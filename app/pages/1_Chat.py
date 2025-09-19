import os
import sys
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.config import settings
from src.modules.chunking import DefaultTextSplitter
from src.modules.embeddings import OpenAIEmbeddings
from src.modules.vectorstore import ChromaVectorStore, ElasticVectorStore, CompositeVectorStore
from src.modules.retriever import SimpleRetriever, HybridRetriever
from src.modules.llm import OpenAIChatLLM
from src.modules.pipeline import RAGPipeline
from src.modules.tracing import TraceRecorder

st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="wide")

st.title("💬 RAG Chat")
if not settings.openai_api_key:
    st.warning(".env의 OPENAI_API_KEY가 필요합니다.")

@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    splitter = DefaultTextSplitter()
    embeddings = OpenAIEmbeddings()
    dense = ChromaVectorStore(embeddings=embeddings)
    sparse = ElasticVectorStore()
    store = CompositeVectorStore([dense, sparse])
    # ES 미사용/미가용 시 HybridRetriever가 자동 폴백
    retriever = HybridRetriever(dense_store=dense, sparse_store=sparse)
    llm = OpenAIChatLLM()
    return RAGPipeline(splitter, embeddings, store, retriever, llm)


with st.sidebar:
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=settings.top_k)
    max_tokens = st.slider("Max Tokens", min_value=128, max_value=2048, value=settings.max_tokens, step=64)
    st.markdown("---")
    enable_rerank = st.toggle("Rerank 활성화", value=settings.rerank_enabled)
    rerank_top_n = st.slider("Rerank Top-N", min_value=1, max_value=10, value=min(settings.rerank_top_n, settings.top_k))

pipeline = get_pipeline()

query = st.text_input("질문을 입력하세요")
if st.button("질의") and query.strip():
    trace = TraceRecorder()
    with st.spinner("검색 및 생성 중..."):
        # 동적 설정 반영 (세션 동안)
        from src.config import settings as _s
        _s.rerank_enabled = bool(enable_rerank)
        _s.rerank_top_n = int(min(rerank_top_n, top_k))
        result = pipeline.answer(query, k=top_k, trace=trace, max_tokens=max_tokens)

    st.subheader("응답")
    st.write(result["answer"])  # type: ignore

    st.subheader("참조 문서")
    for i, meta in enumerate(result["sources"]):  # type: ignore
        src = meta.get('source', 'unknown')
        law_path = meta.get('law_path')
        if law_path:
            st.markdown(f"- 소스 {i+1}: `{src}` · 법령 위치: {law_path}")
        else:
            st.markdown(f"- 소스 {i+1}: `{src}`")

    # 단계 트레이스
    with st.expander("단계 트레이스 보기"):
        events = result["trace"]  # type: ignore
        for e in events:
            st.json(e)


