from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

from .types import Document


@dataclass
class EvalItem:
    question: str
    ground_truth_answer: Optional[str]
    ground_truth_sources: List[str]
    type: Optional[str] = None


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    mrr_at_k: float


@dataclass
class GenerationMetrics:
    relevance_cosine: Optional[float]
    bleu: Optional[float]
    rouge_l: Optional[float]
    bertscore_f1: Optional[float]
    answer_length: int
    ground_truth_length: int
    faithfulness_judgement: Optional[float]


@dataclass
class LatencyMetrics:
    end_to_end_ms: float
    retriever_ms: float
    generator_ms: float


@dataclass
class PerItemResult:
    question: str
    answer: str
    retrieved_sources: List[str]
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    latency: LatencyMetrics
    meta: Dict[str, Any]


@dataclass
class AggregateResult:
    num_items: int
    retrieval_avg: RetrievalMetrics
    generation_avg: GenerationMetrics
    latency_avg: LatencyMetrics
    per_type: Optional[Dict[str, Dict[str, Any]]] = None


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        return 0.0
    import math

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    na = math.sqrt(sum(a * a for a in vec_a))
    nb = math.sqrt(sum(b * b for b in vec_b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _compute_bleu(hyp: str, ref: str) -> float:
    try:
        import sacrebleu

        score = sacrebleu.corpus_bleu([hyp], [[ref]])
        return float(score.score) / 100.0
    except Exception:
        return 0.0


def _compute_rouge_l(hyp: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(ref, hyp)
        return float(scores["rougeL"].fmeasure)
    except Exception:
        return 0.0


def _compute_bertscore_f1(hyp: str, ref: str, model_type: str = "xlm-roberta-large") -> float:
    try:
        from bert_score import score as bertscore

        P, R, F1 = bertscore([hyp], [ref], lang="ko", model_type=model_type)
        return float(F1.mean().item())
    except Exception:
        return 0.0


def compute_retrieval_metrics(
    retrieved_sources: Sequence[str],
    ground_truth_sources: Sequence[str],
) -> RetrievalMetrics:
    retrieved_set = list(retrieved_sources)
    gt_set = set(s.strip() for s in ground_truth_sources if s and str(s).strip())
    if not retrieved_set:
        return RetrievalMetrics(recall_at_k=0.0, precision_at_k=0.0, mrr_at_k=0.0)

    # 정답 매칭은 포함 관계(파일 경로/식별자 일부 문자열 포함)로 간주
    def is_match(src: str) -> bool:
        return any((gt in src) or (src in gt) for gt in gt_set)

    hits = [1 if is_match(src) else 0 for src in retrieved_set]
    num_hits = sum(hits)
    k = len(retrieved_set)

    # Recall: 정답 소스 중 몇 개가 top-k에 포함되었는가
    recall = 0.0
    if gt_set:
        recall = min(num_hits / len(gt_set), 1.0)

    # Precision: top-k 중 관련 있는 비율
    precision = num_hits / k if k > 0 else 0.0

    # MRR: 첫 관련 문서의 역순위 평균
    mrr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1.0 / rank
            break

    return RetrievalMetrics(recall_at_k=recall, precision_at_k=precision, mrr_at_k=mrr)


def compute_generation_metrics(
    answer: str,
    ground_truth_answer: Optional[str],
    embeddings: Any,
    judge_score: Optional[float] = None,
    enable_bleu_rouge: bool = True,
    enable_bertscore: bool = False,
    bert_model_type: str = "xlm-roberta-large",
) -> GenerationMetrics:
    # 임베딩 코사인 유사도 기반 간단 Relevance 점수 (정답 텍스트가 있는 경우)
    rel = None
    bleu = None
    rouge_l = None
    bert_f1 = None
    if ground_truth_answer:
        try:
            a_vec = embeddings.embed_query(answer)
            g_vec = embeddings.embed_query(ground_truth_answer)
            rel = _cosine_similarity(a_vec, g_vec)
        except Exception:
            rel = None
        if enable_bleu_rouge:
            try:
                bleu = _compute_bleu(answer, ground_truth_answer)
            except Exception:
                bleu = None
            try:
                rouge_l = _compute_rouge_l(answer, ground_truth_answer)
            except Exception:
                rouge_l = None
        if enable_bertscore:
            try:
                bert_f1 = _compute_bertscore_f1(answer, ground_truth_answer, model_type=bert_model_type)
            except Exception:
                bert_f1 = None
    return GenerationMetrics(
        relevance_cosine=rel,
        bleu=bleu,
        rouge_l=rouge_l,
        bertscore_f1=bert_f1,
        answer_length=len(answer or ""),
        ground_truth_length=len(ground_truth_answer or ""),
        faithfulness_judgement=judge_score,
    )


def judge_faithfulness_with_llm(
    llm: Any,
    answer: str,
    retrieved_docs: Sequence[Document],
) -> Optional[float]:
    try:
        context = "\n\n".join(d.page_content for d in retrieved_docs[:4])
        prompt = (
            "다음 답변이 주어진 문서 컨텍스트에 근거하는지 0~1 사이 점수로 평가하세요.\n"
            "0은 전혀 근거 없음, 1은 완전히 근거함.\n"
            "컨텍스트:\n" + context + "\n\n답변:\n" + answer + "\n\n점수만 숫자로 출력"
        )
        # ChatOpenAI 등 LC LLM 호환 처리: predict 우선, 없으면 invoke 사용 후 content 추출
        if hasattr(llm, "predict"):
            text = llm.predict(prompt)
        elif hasattr(llm, "invoke"):
            msg = llm.invoke(prompt)
            text = getattr(msg, "content", str(msg))
        else:
            # 마지막 수단: 함수형 호출 지원 시
            resp = llm(prompt)
            text = getattr(resp, "content", str(resp))

        # 숫자 파싱
        import re

        m = re.search(r"\d*\.?\d+", text)
        if not m:
            return None
        score = float(m.group(0))
        score = max(0.0, min(1.0, score))
        return score
    except Exception:
        return None


def run_evaluation(
    items: Sequence[EvalItem],
    retriever: Any,
    llm: Any,
    embeddings: Any,
    top_k: int = 4,
    enable_llm_judge: bool = False,
    enable_bleu_rouge: bool = True,
    enable_bertscore: bool = False,
    bert_model_type: str = "xlm-roberta-large",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    on_item_result: Optional[Callable[[PerItemResult, int, int], None]] = None,
) -> Tuple[List[PerItemResult], AggregateResult]:
    results: List[PerItemResult] = []

    agg_recall = 0.0
    agg_precision = 0.0
    agg_mrr = 0.0
    agg_rel = 0.0
    num_rel = 0
    agg_bleu = 0.0
    num_bleu = 0
    agg_rouge = 0.0
    num_rouge = 0
    agg_bert = 0.0
    num_bert = 0

    agg_e2e = 0.0
    agg_ret = 0.0
    agg_gen = 0.0
    agg_ans_len = 0
    agg_gt_len = 0

    total = len(items)

    # 타입별 집계 버퍼
    type_buf: Dict[str, Dict[str, Any]] = {}
    if progress_callback is not None:
        try:
            progress_callback(0, total)
        except Exception:
            pass

    for idx, it in enumerate(items, start=1):
        if stop_check is not None and stop_check():
            break
        t0 = time.perf_counter()

        # Retrieval
        tr0 = time.perf_counter()
        docs = retriever.get_relevant_documents(it.question, k=top_k)
        tr1 = time.perf_counter()
        retrieved_sources = [str(d.metadata.get("source", "")) for d in docs]
        r_metrics = compute_retrieval_metrics(retrieved_sources, it.ground_truth_sources)

        # Generation
        tg0 = time.perf_counter()
        # llm.generate(question, docs, max_tokens) 사용 (우리 인터페이스)
        try:
            answer = llm.generate(it.question, docs, max_tokens=None)
        except TypeError:
            answer = llm.generate(it.question, docs)  # 백워드 호환
        judge_score = None
        if enable_llm_judge:
            judge_score = judge_faithfulness_with_llm(llm=getattr(llm, "llm", llm), answer=answer, retrieved_docs=docs)
        g_metrics = compute_generation_metrics(
            answer,
            it.ground_truth_answer,
            embeddings,
            judge_score,
            enable_bleu_rouge=enable_bleu_rouge,
            enable_bertscore=enable_bertscore,
            bert_model_type=bert_model_type,
        )
        tg1 = time.perf_counter()

        t1 = time.perf_counter()

        lat = LatencyMetrics(
            end_to_end_ms=(t1 - t0) * 1000.0,
            retriever_ms=(tr1 - tr0) * 1000.0,
            generator_ms=(tg1 - tg0) * 1000.0,
        )

        per = PerItemResult(
            question=it.question,
            answer=answer,
            retrieved_sources=retrieved_sources,
            retrieval=r_metrics,
            generation=g_metrics,
            latency=lat,
            meta={"gt_sources": it.ground_truth_sources, "type": getattr(it, "type", None)},
        )
        results.append(per)
        if on_item_result is not None:
            try:
                on_item_result(per, idx, total)
            except Exception:
                pass

        agg_recall += r_metrics.recall_at_k
        agg_precision += r_metrics.precision_at_k
        agg_mrr += r_metrics.mrr_at_k
        if g_metrics.relevance_cosine is not None:
            agg_rel += g_metrics.relevance_cosine
            num_rel += 1
        if g_metrics.bleu is not None:
            agg_bleu += g_metrics.bleu
            num_bleu += 1
        if g_metrics.rouge_l is not None:
            agg_rouge += g_metrics.rouge_l
            num_rouge += 1
        if g_metrics.bertscore_f1 is not None:
            agg_bert += g_metrics.bertscore_f1
            num_bert += 1
        agg_e2e += lat.end_to_end_ms
        agg_ret += lat.retriever_ms
        agg_gen += lat.generator_ms
        agg_ans_len += g_metrics.answer_length
        agg_gt_len += g_metrics.ground_truth_length

        # 타입별 누적 (type이 있는 항목만)
        tkey = getattr(it, "type", None)
        if tkey:
            buf = type_buf.setdefault(tkey, {
                "count": 0,
                "recall": 0.0,
                "precision": 0.0,
                "mrr": 0.0,
                "rel": 0.0,
                "num_rel": 0,
                "bleu": 0.0,
                "num_bleu": 0,
                "rouge": 0.0,
                "num_rouge": 0,
                "bert": 0.0,
                "num_bert": 0,
                "e2e": 0.0,
                "ret": 0.0,
                "gen": 0.0,
                "ans_len": 0,
                "gt_len": 0,
            })
            buf["count"] += 1
            buf["recall"] += r_metrics.recall_at_k
            buf["precision"] += r_metrics.precision_at_k
            buf["mrr"] += r_metrics.mrr_at_k
            if g_metrics.relevance_cosine is not None:
                buf["rel"] += g_metrics.relevance_cosine
                buf["num_rel"] += 1
            if g_metrics.bleu is not None:
                buf["bleu"] += g_metrics.bleu
                buf["num_bleu"] += 1
            if g_metrics.rouge_l is not None:
                buf["rouge"] += g_metrics.rouge_l
                buf["num_rouge"] += 1
            if g_metrics.bertscore_f1 is not None:
                buf["bert"] += g_metrics.bertscore_f1
                buf["num_bert"] += 1
            buf["e2e"] += lat.end_to_end_ms
            buf["ret"] += lat.retriever_ms
            buf["gen"] += lat.generator_ms
            buf["ans_len"] += g_metrics.answer_length
            buf["gt_len"] += g_metrics.ground_truth_length

        if progress_callback is not None:
            try:
                progress_callback(idx, total)
            except Exception:
                pass

    n = max(1, len(items))
    retrieval_avg = RetrievalMetrics(
        recall_at_k=agg_recall / n,
        precision_at_k=agg_precision / n,
        mrr_at_k=agg_mrr / n,
    )
    generation_avg = GenerationMetrics(
        relevance_cosine=(agg_rel / num_rel) if num_rel > 0 else None,
        bleu=(agg_bleu / num_bleu) if num_bleu > 0 else None,
        rouge_l=(agg_rouge / num_rouge) if num_rouge > 0 else None,
        bertscore_f1=(agg_bert / num_bert) if num_bert > 0 else None,
        answer_length=int(agg_ans_len / n),
        ground_truth_length=int(agg_gt_len / n),
        faithfulness_judgement=None,
    )
    latency_avg = LatencyMetrics(
        end_to_end_ms=agg_e2e / n,
        retriever_ms=agg_ret / n,
        generator_ms=agg_gen / n,
    )

    # 타입별 평균 산출
    per_type_out: Dict[str, Dict[str, Any]] = {}
    for tkey, buf in type_buf.items():
        cnt = max(1, int(buf.get("count", 0)))
        retrieval_avg = RetrievalMetrics(
            recall_at_k=buf["recall"] / cnt,
            precision_at_k=buf["precision"] / cnt,
            mrr_at_k=buf["mrr"] / cnt,
        )
        generation_avg = GenerationMetrics(
            relevance_cosine=(buf["rel"] / buf["num_rel"]) if buf["num_rel"] > 0 else None,
            bleu=(buf["bleu"] / buf["num_bleu"]) if buf["num_bleu"] > 0 else None,
            rouge_l=(buf["rouge"] / buf["num_rouge"]) if buf["num_rouge"] > 0 else None,
            bertscore_f1=(buf["bert"] / buf["num_bert"]) if buf["num_bert"] > 0 else None,
            answer_length=int(buf["ans_len"] / cnt),
            ground_truth_length=int(buf["gt_len"] / cnt),
            faithfulness_judgement=None,
        )
        latency_avg = LatencyMetrics(
            end_to_end_ms=buf["e2e"] / cnt,
            retriever_ms=buf["ret"] / cnt,
            generator_ms=buf["gen"] / cnt,
        )
        per_type_out[tkey] = {
            "num_items": int(buf["count"]),
            "retrieval_avg": asdict(retrieval_avg),
            "generation_avg": asdict(generation_avg),
            "latency_avg": asdict(latency_avg),
        }

    aggregate = AggregateResult(
        num_items=len(items),
        retrieval_avg=retrieval_avg,
        generation_avg=generation_avg,
        latency_avg=latency_avg,
        per_type=per_type_out or None,
    )

    return results, aggregate


def parse_dataset(file_bytes: bytes, filename: str) -> List[EvalItem]:
    # memoryview/bytearray 등도 허용
    if isinstance(file_bytes, memoryview):
        data = file_bytes.tobytes()
    elif isinstance(file_bytes, bytearray):
        data = bytes(file_bytes)
    else:
        data = file_bytes
    name = filename.lower()
    if name.endswith(".jsonl") or name.endswith(".ndjson"):
        lines = data.decode("utf-8", errors="ignore").splitlines()
        items: List[EvalItem] = []
        for line in lines:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(_to_eval_item(obj))
        return items
    elif name.endswith(".json"):
        obj = json.loads(data.decode("utf-8", errors="ignore"))
        if isinstance(obj, list):
            return [_to_eval_item(o) for o in obj]
        raise ValueError("JSON 파일은 리스트 형태여야 합니다.")
    elif name.endswith(".csv"):
        import csv
        text = data.decode("utf-8", errors="ignore")
        reader = csv.DictReader(text.splitlines())
        items2: List[EvalItem] = []
        for row in reader:
            items2.append(_to_eval_item(row))
        return items2
    else:
        raise ValueError("지원하지 않는 형식입니다. csv, json, jsonl을 사용하세요.")


def _to_eval_item(obj: Dict[str, Any]) -> EvalItem:
    q = str(obj.get("question", "")).strip()
    gt_a = obj.get("ground_truth_answer")
    if gt_a is not None:
        gt_a = str(gt_a)

    raw_src = obj.get("ground_truth_sources")
    sources: List[str] = []
    if isinstance(raw_src, list):
        sources = [str(s) for s in raw_src]
    elif isinstance(raw_src, str):
        # 세미콜론/쉼표 구분 문자열 허용
        parts = [p.strip() for p in raw_src.replace(";", ",").split(",")]
        sources = [p for p in parts if p]
    t = obj.get("type")
    if t is not None:
        t = str(t).strip() or None
    return EvalItem(question=q, ground_truth_answer=gt_a, ground_truth_sources=sources, type=t)


def to_json_report(results: Sequence[PerItemResult], aggregate: AggregateResult, config: Optional[Dict[str, Any]] = None) -> str:
    data = {
        "config": config or {},
        "aggregate": {
            "num_items": aggregate.num_items,
            "retrieval_avg": asdict(aggregate.retrieval_avg),
            "generation_avg": asdict(aggregate.generation_avg),
            "latency_avg": asdict(aggregate.latency_avg),
        },
        "items": [
            {
                "question": r.question,
                "answer": r.answer,
                "retrieved_sources": r.retrieved_sources,
                "retrieval": asdict(r.retrieval),
                "generation": asdict(r.generation),
                "latency": asdict(r.latency),
                "meta": r.meta,
            }
            for r in results
        ],
    }
    if aggregate.per_type:
        data["aggregate"]["per_type"] = aggregate.per_type
    return json.dumps(data, ensure_ascii=False, indent=2)


DATASET_TEMPLATE_JSONL = (
    """\
{"question": "도심항공교통 규칙의 목적은 무엇인가?", "ground_truth_answer": "이 규칙은 도심항공교통 활용 촉진 및 지원에 관한 법률 및 같은 법 시행령에서 위임된 사항과 그 시행에 필요한 사항을 규정함을 목적으로 한다.", "ground_truth_sources": ["HJ_B_014648.csv"]}
{"question": "버티포트에 필수적으로 설치해야 하는 시설에는 무엇이 있나?", "ground_truth_answer": "버티포트의 필수시설에는 풍향지시기, 버티포트 식별표지가 있다.", "ground_truth_sources": ["HJ_B_014648.csv"]}
{"question": "시범운용에서 규제특례가 적용되는 업무에는 어떤 것들이 있나?", "ground_truth_answer": "공중광고, 사진 또는 영상 촬영, 산림·관로·전선 등의 순찰, 치안 활동, 도심형항공기사용사업에 따른 업무 등이 규제특례 적용 대상이다.", "ground_truth_sources": ["HJ_B_014648.csv"]}
{"question": "도심형항공기사용사업의 범위에는 어떤 업무가 포함되나?", "ground_truth_answer": "수색·구조·의료·응급후송, 도심형항공기를 사용하는 비행훈련, 제3조제1호~제4호의 업무, 국토교통부장관이 인정하는 업무가 포함된다.", "ground_truth_sources": ["HJ_B_014648.csv"]}
{"question": "실증사업구역을 지정한 경우 국토교통부장관이 관보에 고시해야 하는 사항은?", "ground_truth_answer": "실증사업구역의 명칭·위치·범위, 지정목적, 지정 연월일, 지정기간, 기타 필요한 사항을 관보에 고시해야 한다.", "ground_truth_sources": ["HJ_B_014648.csv"]}
{"question": "국가유산 기본법의 목적은 무엇인가?", "ground_truth_answer": "국가와 지방자치단체의 책임을 명확히 하여 국가유산을 보호하고 창조적으로 계승함으로써 국민의 삶의 질 향상을 도모하는 데 목적이 있다.", "ground_truth_sources": ["HJ_B_014648.csv"]}
{"question": "국가유산 기본법의 기본이념은 무엇인가?", "ground_truth_answer": "국가유산을 삶의 뿌리이자 창의성의 원천으로 인식하고, 그 가치를 지키고 향유하며 미래 세대에 가치 있게 전하는 것을 기본이념으로 한다.", "ground_truth_sources": ["HJ_B_014648.csv"]}
{"question": "이 법에서 '국가유산'은 무엇을 의미하나?", "ground_truth_answer": "인위적·자연적으로 형성된 역사적, 예술적, 학술적 또는 경관적 가치가 큰 문화유산, 자연유산, 무형유산을 말한다.", "ground_truth_sources": ["HJ_B_014648.csv"]}
{"question": "문화유산은 어떻게 정의되나?", "ground_truth_answer": "우리 역사와 전통의 산물로서 문화의 고유성과 정체성을 나타내는 유형의 문화적 유산을 의미한다.", "ground_truth_sources": ["HJ_B_014648.csv"]}
{"question": "자연유산에는 어떤 것들이 포함되나?", "ground_truth_answer": "지형, 지질, 생태계, 생물종 등 자연적으로 형성된 경관적 가치가 큰 요소들이 포함된다.", "ground_truth_sources": ["HJ_B_014648.csv"]}

"""
)


