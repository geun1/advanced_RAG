from typing import Any, Dict, List, Protocol, Sequence
from langchain_core.documents import Document


class TextSplitter(Protocol):
    def split_text(self, text: str) -> List[str]:
        ...


class Embeddings(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        ...
    def embed_query(self, text: str) -> List[float]:
        ...


class VectorStore(Protocol):
    def add_texts(self, texts: Sequence[str], metadatas: Sequence[Dict[str, Any]]) -> None:
        ...

    def similarity_search(self, query: str, k: int) -> List[Document]:
        ...


class Retriever(Protocol):
    def get_relevant_documents(self, query: str, k: int) -> List[Document]:
        ...


class ChatLLM(Protocol):
    def generate(self, question: str, context_docs: Sequence[Document], max_tokens: int) -> str:
        ...


