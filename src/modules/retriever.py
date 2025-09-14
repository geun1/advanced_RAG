from typing import List, Optional

from ..config import settings
from .types import Document, Retriever, VectorStore


class SimpleRetriever(Retriever):
    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        return self.store.similarity_search(query, k or settings.top_k)


