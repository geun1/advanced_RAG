import os
import sys
from pathlib import Path
import streamlit as st
import json
import uuid
import threading
import time
from datetime import datetime, timezone, timedelta

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.config import settings
from src.modules.embeddings import OpenAIEmbeddings
from src.modules.vectorstore import ChromaVectorStore
from src.modules.retriever import SimpleRetriever
from src.modules.llm import OpenAIChatLLM
from src.modules.evaluation import (
    parse_dataset,
    run_evaluation,
    to_json_report,
    DATASET_TEMPLATE_JSONL,
)
from src.modules.utils import ensure_dir


st.set_page_config(page_title="RAG Evaluation", page_icon="📈", layout="wide")
st.title("📈 RAG 성능 평가")

st.markdown("""
다음 형식(csv/json/jsonl)의 평가 데이터셋을 업로드하세요. 각 항목에는 최소 `question`이 필요합니다.
- `ground_truth_answer`: 선택(정답 텍스트) — 있으면 의미 유사도(relevance) 계산
- `ground_truth_sources`: 선택(문서 식별자 목록 또는 쉼표/세미콜론 구분 문자열) — Recall/MRR/Precision 계산에 사용
""")

with st.expander("데이터셋 템플릿 보기"):
    st.code(DATASET_TEMPLATE_JSONL, language="json")


@st.cache_resource(show_spinner=False)
def get_components():
    embeddings = OpenAIEmbeddings()
    store = ChromaVectorStore(embeddings=embeddings)
    retriever = SimpleRetriever(store)
    llm = OpenAIChatLLM()
    return embeddings, retriever, llm


embeddings, retriever, llm = get_components()

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    uploaded = st.file_uploader("평가 데이터셋 업로드 (csv/json/jsonl)")
with col2:
    top_k = st.slider("Top-K", 1, 10, settings.top_k)
with col3:
    use_llm_judge = st.checkbox("LLM 판사 사용(근거 충실도)", value=False)

col4, col5 = st.columns([1, 1])
with col4:
    use_bleu_rouge = st.checkbox("BLEU/ROUGE 사용", value=True)
with col5:
    use_bertscore = st.checkbox("BERTScore 사용", value=False)

bert_model_type = "xlm-roberta-large"
if use_bertscore:
    bert_model_type = st.text_input("BERTScore 모델", value="xlm-roberta-large")

items = []
if uploaded is not None:
    try:
        # memoryview가 아닌 bytes로 전달
        items = parse_dataset(uploaded.getvalue(), uploaded.name)
        st.success(f"항목 {len(items)}개 로드")
    except Exception as e:
        st.error(f"데이터셋 파싱 실패: {e}")

# =============================
# 백그라운드 평가 관리자
# =============================

RUNS_DIR = BASE_DIR / "data" / "eval_runs"
ensure_dir(str(RUNS_DIR))


class EvalManager:
    def __init__(self) -> None:
        self._threads = {}
        self._stops = {}
        ensure_dir(str(RUNS_DIR))

    def _run_dir(self, run_id: str) -> Path:
        return RUNS_DIR / run_id

    def list_runs(self):
        runs = []
        for p in sorted(RUNS_DIR.glob("*")):
            if not p.is_dir():
                continue
            status_path = p / "status.json"
            meta = {"run_id": p.name, "status": "unknown"}
            if status_path.exists():
                try:
                    meta.update(json.loads(status_path.read_text(encoding="utf-8")))
                except Exception:
                    pass
            runs.append(meta)
        runs.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        return runs

    def stop(self, run_id: str) -> None:
        self._stops[run_id] = True
        # 파일 신호로도 중지 플래그 남김(세션 재생성 대비)
        try:
            (self._run_dir(run_id) / "STOP").write_text("1", encoding="utf-8")
        except Exception:
            pass

    def is_running(self, run_id: str) -> bool:
        t = self._threads.get(run_id)
        return t.is_alive() if t else False

    def start(self, items, params: dict) -> str:
        run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        run_path = self._run_dir(run_id)
        ensure_dir(str(run_path))
        (run_path / "results.jsonl").write_text("", encoding="utf-8")
        (run_path / "config.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
        status = {
            "run_id": run_id,
            "status": "running",
            "done": 0,
            "total": len(items),
            "started_at": datetime.utcnow().isoformat() + "Z",
        }
        (run_path / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

        self._stops[run_id] = False

        def on_progress(done: int, total: int) -> None:
            status_path = run_path / "status.json"
            s = {
                "run_id": run_id,
                "status": "running",
                "done": done,
                "total": total,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            status_path.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

        def on_item(res, idx: int, total: int) -> None:
            line = json.dumps({
                "question": res.question,
                "answer": res.answer,
                "retrieved_sources": res.retrieved_sources,
                "retrieval": {
                    "recall_at_k": res.retrieval.recall_at_k,
                    "precision_at_k": res.retrieval.precision_at_k,
                    "mrr_at_k": res.retrieval.mrr_at_k,
                },
                "generation": {
                    "relevance_cosine": res.generation.relevance_cosine,
                    "bleu": res.generation.bleu,
                    "rouge_l": res.generation.rouge_l,
                    "bertscore_f1": res.generation.bertscore_f1,
                    "faithfulness_judgement": res.generation.faithfulness_judgement,
                    "answer_length": res.generation.answer_length,
                    "ground_truth_length": res.generation.ground_truth_length,
                },
                "latency": {
                    "end_to_end_ms": res.latency.end_to_end_ms,
                    "retriever_ms": res.latency.retriever_ms,
                    "generator_ms": res.latency.generator_ms,
                },
                "meta": res.meta,
                "idx": idx,
                "total": total,
            }, ensure_ascii=False)
            with (run_path / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        def stop_check() -> bool:
            # 메모리 플래그 또는 STOP 파일 존재 시 중지
            if self._stops.get(run_id, False):
                return True
            try:
                return (run_path / "STOP").exists()
            except Exception:
                return False

        def worker():
            try:
                results, aggregate = run_evaluation(
                    items=items,
                    retriever=retriever,
                    llm=llm,
                    embeddings=embeddings,
                    top_k=params.get("top_k"),
                    enable_llm_judge=params.get("use_llm_judge"),
                    enable_bleu_rouge=params.get("use_bleu_rouge"),
                    enable_bertscore=params.get("use_bertscore"),
                    bert_model_type=params.get("bert_model_type"),
                    progress_callback=on_progress,
                    # on_item_result/stop_check는 evaluation.py에 추가 필요
                    # 아래 두 파라미터는 존재하지 않으면 무시됩니다.
                    on_item_result=on_item,
                    stop_check=stop_check,
                )
                (run_path / "aggregate.json").write_text(json.dumps({
                    "config": params,
                    "num_items": aggregate.num_items,
                    "retrieval_avg": {
                        "recall_at_k": aggregate.retrieval_avg.recall_at_k,
                        "precision_at_k": aggregate.retrieval_avg.precision_at_k,
                        "mrr_at_k": aggregate.retrieval_avg.mrr_at_k,
                    },
                    "generation_avg": {
                        "relevance_cosine": aggregate.generation_avg.relevance_cosine,
                        "bleu": aggregate.generation_avg.bleu,
                        "rouge_l": aggregate.generation_avg.rouge_l,
                        "bertscore_f1": aggregate.generation_avg.bertscore_f1,
                        "answer_length": aggregate.generation_avg.answer_length,
                        "ground_truth_length": aggregate.generation_avg.ground_truth_length,
                    },
                    "latency_avg": {
                        "end_to_end_ms": aggregate.latency_avg.end_to_end_ms,
                        "retriever_ms": aggregate.latency_avg.retriever_ms,
                        "generator_ms": aggregate.latency_avg.generator_ms,
                    },
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                status = {
                    "run_id": run_id,
                    "status": "completed" if not stop_check() else "stopped",
                    "done": len(results),
                    "total": len(items),
                    "finished_at": datetime.utcnow().isoformat() + "Z",
                }
                (run_path / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:
                (run_path / "status.json").write_text(json.dumps({
                    "run_id": run_id,
                    "status": "error",
                    "error": str(e),
                }, ensure_ascii=False, indent=2), encoding="utf-8")

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        self._threads[run_id] = th
        return run_id


@st.cache_resource(show_spinner=False)
def get_eval_manager() -> EvalManager:
    return EvalManager()


st.divider()
st.subheader("백그라운드 성능 평가")

mngr = get_eval_manager()

bg_items = items  # 업로드한 동일 items 재사용
if bg_items:
    # 항상 백그라운드 실행
    if st.button("평가 실행"):
        # 실행 시점 설정 수집
        from src.modules.llm import OpenAIChatLLM
        from src.modules.embeddings import OpenAIEmbeddings
        from src.modules.vectorstore import ChromaVectorStore
        from src.config import settings as _s
        # Rerank 설정
        rerank_enabled = bool(_s.rerank_enabled)
        rerank_top_n = int(_s.rerank_top_n)
        rerank_provider = str(_s.rerank_provider)
        rerank_model = None
        if rerank_provider == "cohere":
            rerank_model = getattr(_s, "cohere_rerank_model", None)

        params = {
            "top_k": top_k,
            "max_tokens": _s.max_tokens,
            "use_llm_judge": use_llm_judge,
            "use_bleu_rouge": use_bleu_rouge,
            "use_bertscore": use_bertscore,
            "bert_model_type": bert_model_type,
            # 구성 정보
            "chat_model": getattr(llm, "model", None),
            "embedding_model": getattr(embeddings, "model", None),
            "vectorstore": "Chroma",
            "vectorstore_collection": getattr(settings, "collection_name", None),
            "rerank_enabled": rerank_enabled,
            "rerank_top_n": rerank_top_n,
            "rerank_provider": rerank_provider,
            "rerank_model": rerank_model,
            "retriever": "SimpleRetriever",
        }
        run_id = mngr.start(bg_items, params)
        st.success(f"백그라운드 실행 시작: {run_id}")

st.markdown("### 실행 중인 평가")
running = [r for r in mngr.list_runs() if r.get("status") == "running"]
# 자동 새로고침: 실행 중 작업이 있으면 3초 간격으로 재실행
if running:
    last = st.session_state.get("bg_last_refresh_ts", 0.0)
    now = time.time()
    if now - last > 3.0:
        st.session_state["bg_last_refresh_ts"] = now
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()  # for older versions
            except Exception:
                pass
if not running:
    st.write("실행 중인 작업이 없습니다.")
else:
    for r in running:
        cols = st.columns([3, 2, 2, 2, 2])
        with cols[0]:
            st.write(r.get("run_id"))
        with cols[1]:
            done = int(r.get("done", 0)); total = int(r.get("total", 0)) or 1
            st.write(f"{done}/{total}")
        with cols[2]:
            st.progress(int(done/total*100))
        with cols[3]:
            rk = r.get("run_id")
            if st.button("중지", key=f"stop-{rk}"):
                st.session_state[f"confirm_stop_{rk}"] = True
            if st.session_state.get(f"confirm_stop_{rk}"):
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("확인", key=f"ok-{rk}"):
                        mngr.stop(rk)
                        st.session_state[f"confirm_stop_{rk}"] = False
                        st.rerun()
                with c2:
                    if st.button("취소", key=f"cancel-{rk}"):
                        st.session_state[f"confirm_stop_{rk}"] = False
        with cols[4]:
            st.write(r.get("status"))

st.markdown("### 과거 평가 기록")
history = [r for r in mngr.list_runs() if r.get("status") in ("completed", "stopped", "error")]
if not history:
    st.write("기록이 없습니다.")
else:
    for r in history[:20]:
        run_id = r.get('run_id')
        status_label = r.get('status')
        with st.expander(f"{run_id} · {status_label}"):
            run_dir = RUNS_DIR / run_id
            agg_path = run_dir / "aggregate.json"
            res_path = run_dir / "results.jsonl"
            status_path = run_dir / "status.json"

            # 한국시간 표시
            started_at_kst = None
            finished_at_kst = None
            duration_text = None
            try:
                if status_path.exists():
                    sdata = json.loads(status_path.read_text(encoding="utf-8"))
                    s = sdata.get("started_at")
                    f = sdata.get("finished_at")
                    def to_kst(ts: str):
                        try:
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            return dt.astimezone(timezone(timedelta(hours=9)))
                        except Exception:
                            return None
                    sdt = to_kst(s) if s else None
                    fdt = to_kst(f) if f else None
                    if sdt:
                        started_at_kst = sdt.strftime("%Y-%m-%d %H:%M:%S KST")
                    if fdt:
                        finished_at_kst = fdt.strftime("%Y-%m-%d %H:%M:%S KST")
                    if sdt and fdt:
                        delta = fdt - sdt
                        secs = int(delta.total_seconds())
                        mins, secs = divmod(secs, 60)
                        hrs, mins = divmod(mins, 60)
                        duration_text = f"{hrs:02d}:{mins:02d}:{secs:02d}"
            except Exception:
                pass

            info_cols = st.columns(3)
            with info_cols[0]:
                st.write(f"시작: {started_at_kst or '-'}")
            with info_cols[1]:
                st.write(f"종료: {finished_at_kst or '-'}")
            with info_cols[2]:
                st.write(f"경과: {duration_text or '-'}")

            st.divider()
            if agg_path.exists():
                st.markdown("집계 결과")
                st.json(json.loads(agg_path.read_text(encoding="utf-8")))

            if res_path.exists():
                st.download_button(
                    label="JSONL 다운로드",
                    data=res_path.read_bytes(),
                    file_name=f"{run_id}-results.jsonl",
                    mime="application/json",
                    key=f"dlbtn-{run_id}"
                )

            # 메모 유지 기능
            memo_path = run_dir / "memo.txt"
            existing_memo = ""
            try:
                if memo_path.exists():
                    existing_memo = memo_path.read_text(encoding="utf-8")
            except Exception:
                pass
            memo = st.text_area("메모", value=existing_memo, key=f"memo-{run_id}")
            if st.button("메모 저장", key=f"save-memo-{run_id}"):
                try:
                    memo_path.write_text(memo or "", encoding="utf-8")
                    st.success("메모를 저장했습니다.")
                except Exception as e:
                    st.error(f"메모 저장 실패: {e}")


