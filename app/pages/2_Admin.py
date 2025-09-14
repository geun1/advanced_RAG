import os
import sys
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.config import settings
from src.modules.loaders import load_file_to_document
from src.modules.chunking import DefaultTextSplitter
from src.modules.embeddings import OpenAIEmbeddings
from src.modules.vectorstore import ChromaVectorStore
from src.modules.retriever import SimpleRetriever
from src.modules.llm import OpenAIChatLLM
from src.modules.pipeline import RAGPipeline
from src.modules.utils import ensure_dir
from src.modules.tracing import TraceRecorder

st.set_page_config(page_title="RAG Admin", page_icon="🛠️", layout="wide")
st.title("🛠️ Admin: 문서 업로드 및 색인")

docs_dir = os.path.abspath(os.path.join("data", "docs"))
ensure_dir(docs_dir)

@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    splitter = DefaultTextSplitter()
    embeddings = OpenAIEmbeddings()
    store = ChromaVectorStore(embeddings=embeddings)
    retriever = SimpleRetriever(store)
    llm = OpenAIChatLLM()
    return RAGPipeline(splitter, embeddings, store, retriever, llm)

pipeline = get_pipeline()

uploaded_files = st.file_uploader("문서 업로드 (txt, md, pdf 등)", accept_multiple_files=True)
if uploaded_files:
    saved_paths = []
    for f in uploaded_files:
        save_path = os.path.join(docs_dir, f.name)
        with open(save_path, "wb") as out:
            out.write(f.getbuffer())
        saved_paths.append(save_path)
    st.success(f"{len(saved_paths)}개 파일 저장 완료")

if st.button("색인하기"):
    files = [os.path.join(docs_dir, name) for name in os.listdir(docs_dir)]
    docs = [load_file_to_document(p, {"uploaded": "true"}) for p in files]
    trace = TraceRecorder()
    with st.spinner("청킹 및 벡터DB 반영 중..."):
        stats = pipeline.index_documents(docs, trace=trace)
    st.success(f"색인 완료. 총 청크 수: {stats['total_chunks']}")
    with st.expander("단계 트레이스 보기"):
        for e in trace.as_dicts():
            st.json(e)

if st.button("Chroma 초기화(모든 벡터 삭제)"):
    # Chroma 데이터 디렉터리 삭제
    persist_dir = os.path.abspath(settings.chroma_persist_dir)
    if os.path.exists(persist_dir):
        # 단순 삭제(사용자 책임)
        import shutil
        shutil.rmtree(persist_dir, ignore_errors=True)
        st.warning("Chroma 데이터 초기화 완료. 앱 재시작 후 반영될 수 있습니다.")
    else:
        st.info("초기화할 데이터가 없습니다.")


