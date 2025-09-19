
from typing import Dict, List, Optional, Sequence, Tuple
import hashlib

from ..config import settings
from .types import Document, Retriever, VectorStore


class SimpleRetriever(Retriever):
    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        return self.store.similarity_search(query, k or settings.top_k)



class HybridRetriever(Retriever):
    """Dense(Chroma) + Sparse(Elasticsearch) 앙상블 리트리버.

    - dense_store: 임베딩 유사도 기반
    - sparse_store: BM25 키워드 기반
    - 융합: Reciprocal Rank Fusion (RRF). 점수 = sum(1 / (k0 + rank))
    """

    def __init__(
        self,
        dense_store: VectorStore,
        sparse_store: Optional[VectorStore] = None,
        dense_k: int = 12,
        sparse_k: int = 24,
        rrf_k0: int = 60,
    ) -> None:
        self.dense_store = dense_store
        self.sparse_store = sparse_store
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.rrf_k0 = rrf_k0
        self.last_debug: Dict[str, object] = {}

    def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        top_k = int(k or settings.top_k)
        dense_docs = self._safe_search(self.dense_store, query, self.dense_k)
        sparse_docs: List[Document] = []
        sparse_available = False
        if self.sparse_store is not None:
            sparse_docs = self._safe_search(self.sparse_store, query, self.sparse_k)
            sparse_available = bool(getattr(self.sparse_store, "available", False))
            last_error = getattr(self.sparse_store, "last_error", None)

        if not sparse_docs:
            # 폴백: dense만 사용
            out = dense_docs[:top_k]
            self.last_debug = {
                "strategy": "dense_only",
                "dense_retrieved": len(dense_docs),
                "sparse_retrieved": 0,
                "fused_returned": len(out),
                "es_available": sparse_available,
                "es_last_error": last_error,
                "rrf_k0": self.rrf_k0,
            }
            return out

        fused_docs = self._rrf_fusion(query, [dense_docs, sparse_docs], self.rrf_k0, top_k)
        self.last_debug = {
            "strategy": "hybrid_rrf",
            "dense_retrieved": len(dense_docs),
            "sparse_retrieved": len(sparse_docs),
            "fused_returned": len(fused_docs),
            "es_available": sparse_available,
            "es_last_error": last_error,
            "rrf_k0": self.rrf_k0,
        }
        return fused_docs

    def _safe_search(self, store: VectorStore, query: str, k: int) -> List[Document]:
        try:
            return store.similarity_search(query, k)
        except Exception:
            return []

    def _doc_key(self, d: Document) -> str:
        # 가능한 안정적인 키: source + doc_id + chunk_id + 내용 해시
        src = str(d.metadata.get("source", ""))
        doc_id = str(d.metadata.get("doc_id", ""))
        chunk_id = str(d.metadata.get("chunk_id", ""))
        text = d.page_content or ""
        base = f"{src}|{doc_id}|{chunk_id}|{len(text)}"
        h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
        return f"{base}|{h}"

    def _rrf_fusion(
        self,
        query: str,
        result_lists: Sequence[List[Document]],
        k0: int,
        top_k: int,
    ) -> List[Document]:
        scores: Dict[str, float] = {}
        rep: Dict[str, Document] = {}

        for result in result_lists:
            for rank, doc in enumerate(result, start=1):
                key = self._doc_key(doc)
                rep[key] = doc
                scores[key] = scores.get(key, 0.0) + 1.0 / (k0 + float(rank))

        # 점수 내림차순 정렬 후 상위 top_k 반환
        sorted_items: List[Tuple[str, float]] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out: List[Document] = []
        for key, _score in sorted_items[:top_k]:
            out.append(rep[key])
        return out

