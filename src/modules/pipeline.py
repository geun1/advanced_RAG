from typing import Dict, List, Sequence, Optional

from ..config import settings
from .types import ChatLLM, Document, Embeddings, Retriever, TextSplitter, VectorStore
from .tracing import TraceRecorder


class RAGPipeline:
    def __init__(
        self,
        splitter: TextSplitter,
        embeddings: Embeddings,
        store: VectorStore,
        retriever: Retriever,
        llm: ChatLLM,
    ) -> None:
        self.splitter = splitter
        self.embeddings = embeddings
        self.store = store
        self.retriever = retriever
        self.llm = llm

    # Indexing
    def index_documents(self, docs: Sequence[Document], trace: Optional[TraceRecorder] = None) -> Dict[str, int]:
        trace = trace or TraceRecorder()
        trace.add("index:start", num_docs=len(docs))
        total_chunks = 0
        all_texts: List[str] = []
        all_metas: List[Dict] = []

        for doc_id, doc in enumerate(docs):
            chunks = self.splitter.split_text(doc.page_content)
            trace.add("chunking", source=doc.metadata.get("source", ""), chunks=len(chunks))
            for idx, chunk in enumerate(chunks):
                meta = dict(doc.metadata)
                meta.update({"chunk_id": idx, "doc_id": doc_id})
                all_texts.append(chunk)
                all_metas.append(meta)
                total_chunks += 1

        trace.add("store:add", total_chunks=total_chunks)
        self.store.add_texts(all_texts, all_metas)
        trace.add("index:done", total_chunks=total_chunks)
        return {"total_chunks": total_chunks}

    # Retrieval + Generation
    def answer(self, question: str, k: Optional[int] = None, trace: Optional[TraceRecorder] = None, max_tokens: Optional[int] = None) -> Dict[str, object]:
        trace = trace or TraceRecorder()
        trace.add("retrieval:start", query=question)
        docs = self.retriever.get_relevant_documents(question, k or settings.top_k)
        trace.add("retrieval:done", num_docs=len(docs))
        effective_max_tokens = max_tokens or settings.max_tokens
        answer = self.llm.generate(question, docs, max_tokens=effective_max_tokens)
        trace.add("llm:done", tokens=effective_max_tokens)
        sources = [d.metadata for d in docs]
        return {"answer": answer, "sources": sources, "trace": trace.as_dicts()}


