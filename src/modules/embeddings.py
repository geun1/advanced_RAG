from typing import List, Sequence, Optional
from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings

from ..config import settings
from .types import Embeddings


class OpenAIEmbeddings(Embeddings):
    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or settings.embedding_model
        self.client = LCOpenAIEmbeddings(model=self.model, api_key=settings.openai_api_key)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        return self.client.embed_documents(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self.client.embed_query(text)


