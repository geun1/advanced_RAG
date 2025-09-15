from typing import List, Sequence, Optional

try:
    from cohere import Client as CohereClient  # type: ignore
except Exception:  # pragma: no cover - optional dep
    CohereClient = None  # type: ignore

from langchain_core.documents import Document

from ..config import settings


class BaseReranker:
    def rerank(self, query: str, docs: Sequence[Document], top_n: Optional[int] = None) -> List[Document]:
        raise NotImplementedError


class NoOpReranker(BaseReranker):
    def rerank(self, query: str, docs: Sequence[Document], top_n: Optional[int] = None) -> List[Document]:
        n = top_n or settings.rerank_top_n or len(docs)
        return list(docs)[:n]


class CohereReranker(BaseReranker):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        if CohereClient is None:
            raise RuntimeError("cohere 패키지가 설치되어 있지 않습니다. requirements에 cohere를 추가하세요.")
        key = api_key or settings.cohere_api_key
        if not key:
            raise RuntimeError("COHERE_API_KEY가 필요합니다.")
        self.client = CohereClient(key)
        self.model = model or settings.cohere_rerank_model

    def rerank(self, query: str, docs: Sequence[Document], top_n: Optional[int] = None) -> List[Document]:
        if not docs:
            return []
        inputs = [d.page_content for d in docs]
        n = top_n or settings.rerank_top_n or len(inputs)
        # Cohere rerank API
        res = self.client.rerank(model=self.model, query=query, documents=inputs, top_n=min(n, len(inputs)))
        # res.results는 각 문서의 index와 relevance score 포함
        # 원래 문서 순서에 매핑
        ranked_docs: List[Document] = []
        for item in res.results:  # type: ignore[attr-defined]
            idx = getattr(item, "index", None)
            if idx is None or not (0 <= idx < len(docs)):
                continue
            ranked_docs.append(docs[idx])
        return ranked_docs


def get_reranker() -> BaseReranker:
    if not settings.rerank_enabled:
        return NoOpReranker()
    provider = (settings.rerank_provider or "").lower()
    if provider == "cohere":
        return CohereReranker()
    # 기본은 no-op
    return NoOpReranker()


