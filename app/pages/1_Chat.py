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
from src.modules.vectorstore import ChromaVectorStore
from src.modules.retriever import SimpleRetriever
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
    store = ChromaVectorStore(embeddings=embeddings)
    retriever = SimpleRetriever(store)
    llm = OpenAIChatLLM()
    return RAGPipeline(splitter, embeddings, store, retriever, llm)


with st.sidebar:
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=settings.top_k)
    max_tokens = st.slider("Max Tokens", min_value=128, max_value=2048, value=settings.max_tokens, step=64)

pipeline = get_pipeline()

query = st.text_input("질문을 입력하세요")
if st.button("질의") and query.strip():
    trace = TraceRecorder()
    with st.spinner("검색 및 생성 중..."):
        result = pipeline.answer(query, k=top_k, trace=trace, max_tokens=max_tokens)

    st.subheader("응답")
    st.write(result["answer"])  # type: ignore

    st.subheader("참조 문서")
    for i, meta in enumerate(result["sources"]):  # type: ignore
        st.markdown(f"- 소스 {i+1}: `{meta.get('source','unknown')}`")

    # 단계 트레이스
    with st.expander("단계 트레이스 보기"):
        events = result["trace"]  # type: ignore
        for e in events:
            st.json(e)


