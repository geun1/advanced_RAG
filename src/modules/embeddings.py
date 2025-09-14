from typing import List, Sequence, Optional
from openai import OpenAI

from ..config import settings
from .types import Embeddings


class OpenAIEmbeddings(Embeddings):
    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or settings.embedding_model
        self.client = OpenAI(api_key=settings.openai_api_key)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model,
            input=list(texts)
        )
        return [d.embedding for d in response.data]


