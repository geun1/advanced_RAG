import os
import sys
from pathlib import Path
import streamlit as st

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

run = st.button("평가 실행")

if run:
    if not items:
        st.warning("평가 항목이 없습니다. 템플릿을 참고해 데이터셋을 준비하세요.")
    else:
        with st.spinner("평가 중..."):
            results, aggregate = run_evaluation(
                items=items,
                retriever=retriever,
                llm=llm,
                embeddings=embeddings,
                top_k=top_k,
                enable_llm_judge=use_llm_judge,
                enable_bleu_rouge=use_bleu_rouge,
                enable_bertscore=use_bertscore,
                bert_model_type=bert_model_type,
            )

        st.subheader("집계 결과")
        m1 = aggregate.retrieval_avg
        m2 = aggregate.generation_avg
        m3 = aggregate.latency_avg
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Recall@k", f"{m1.recall_at_k:.3f}")
            st.metric("Precision@k", f"{m1.precision_at_k:.3f}")
            st.metric("MRR@k", f"{m1.mrr_at_k:.3f}")
        with c2:
            st.metric("Relevance(cos)", f"{(m2.relevance_cosine if m2.relevance_cosine is not None else 0.0):.3f}")
            if m2.bleu is not None:
                st.metric("BLEU", f"{m2.bleu:.3f}")
            if m2.rouge_l is not None:
                st.metric("ROUGE-L", f"{m2.rouge_l:.3f}")
            if m2.bertscore_f1 is not None:
                st.metric("BERTScore-F1", f"{m2.bertscore_f1:.3f}")
        with c3:
            st.metric("End-to-End(ms)", f"{m3.end_to_end_ms:.1f}")
            st.metric("Retriever(ms)", f"{m3.retriever_ms:.1f}")
            st.metric("Generator(ms)", f"{m3.generator_ms:.1f}")
        with c4:
            st.metric("Avg Answer Len", f"{m2.answer_length}")
            st.metric("Avg GT Len", f"{m2.ground_truth_length}")

        st.subheader("개별 결과")
        for i, r in enumerate(results, start=1):
            with st.expander(f"[{i}] {r.question}"):
                st.markdown("**Answer**")
                st.write(r.answer)
                st.markdown("**Retrieved Sources**")
                st.write(r.retrieved_sources)
                st.markdown("**Retrieval**")
                st.json({
                    "recall@k": r.retrieval.recall_at_k,
                    "precision@k": r.retrieval.precision_at_k,
                    "mrr@k": r.retrieval.mrr_at_k,
                })
                st.markdown("**Generation**")
                st.json({
                    "relevance_cosine": r.generation.relevance_cosine,
                    "bleu": r.generation.bleu,
                    "rouge_l": r.generation.rouge_l,
                    "bertscore_f1": r.generation.bertscore_f1,
                    "faithfulness_judgement": r.generation.faithfulness_judgement,
                    "answer_length": r.generation.answer_length,
                })
                st.markdown("**Latency (ms)**")
                st.json({
                    "e2e": r.latency.end_to_end_ms,
                    "retriever": r.latency.retriever_ms,
                    "generator": r.latency.generator_ms,
                })

        st.subheader("리포트 내보내기")
        report = to_json_report(results, aggregate)
        st.download_button(
            label="JSON 다운로드",
            data=report.encode("utf-8"),
            file_name="rag_eval_report.json",
            mime="application/json",
        )


