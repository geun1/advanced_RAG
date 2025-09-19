from typing import Dict, List, Sequence, Optional

from ..config import settings
from .types import ChatLLM, Document, Embeddings, Retriever, TextSplitter, VectorStore
from .re_ranker import get_reranker
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
        self.reranker = get_reranker()

    # Indexing
    def index_documents(self, docs: Sequence[Document], trace: Optional[TraceRecorder] = None, splitter_override: Optional[TextSplitter] = None) -> Dict[str, int]:
        trace = trace or TraceRecorder()
        trace.add("index:start", num_docs=len(docs))
        total_chunks = 0
        all_texts: List[str] = []
        all_metas: List[Dict] = []

        splitter = splitter_override or self.splitter
        for doc_id, doc in enumerate(docs):
            # CSV 법령 원천일 경우, CSV 기반 스플리터 우선 적용
            from .chunking import LegalCSVSplitter
            if str(doc.metadata.get("source", "")).lower().endswith(".csv") and doc.metadata.get("doc_type") == "statute":
                max_chars = getattr(self.splitter, 'chunk_size', None)
                csv_splitter = LegalCSVSplitter(max_chars=max_chars)
                if hasattr(csv_splitter, 'split_with_metadata'):
                    chunks, metas_extra = csv_splitter.split_with_metadata(doc.page_content)
                else:
                    chunks = csv_splitter.split_text(doc.page_content)
                    metas_extra = [{} for _ in chunks]
            else:
                if hasattr(splitter, 'split_with_metadata'):
                    chunks, metas_extra = splitter.split_with_metadata(doc.page_content)  # type: ignore
                else:
                    chunks = splitter.split_text(doc.page_content)
                    metas_extra = [{} for _ in chunks]
            trace.add("chunking", source=doc.metadata.get("source", ""), chunks=len(chunks))
            for idx, chunk in enumerate(chunks):
                meta = dict(doc.metadata)
                meta.update({"chunk_id": idx, "doc_id": doc_id})
                # 법령 메타 병합
                extra = metas_extra[idx] if idx < len(metas_extra) else {}
                if extra:
                    for k, v in extra.items():
                        if v is not None:
                            meta[k] = v
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
        # 리트리버 디버그(하이브리드 여부/ES 상태) 기록
        try:
            debug = getattr(self.retriever, "last_debug", None)
            if isinstance(debug, dict) and debug:
                trace.add("retrieval:debug", **debug)  # type: ignore[arg-type]
        except Exception:
            pass
        trace.add("retrieval:done", num_docs=len(docs))
        # Rerank (optional)
        if settings.rerank_enabled and docs:
            trace.add("rerank:start", provider=settings.rerank_provider, top_n=settings.rerank_top_n)
            reranked = self.reranker.rerank(question, docs, top_n=settings.rerank_top_n)
            docs = reranked
            trace.add("rerank:done", num_docs=len(docs))
        effective_max_tokens = max_tokens or settings.max_tokens
        answer = self.llm.generate(question, docs, max_tokens=effective_max_tokens)
        trace.add("llm:done", tokens=effective_max_tokens)
        sources = [d.metadata for d in docs]
        return {"answer": answer, "sources": sources, "trace": trace.as_dicts()}


