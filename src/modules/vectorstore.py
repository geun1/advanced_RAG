from typing import Any, Dict, List, Sequence, Optional
from langchain_community.vectorstores import Chroma

from ..config import settings
from .utils import ensure_dir
from .types import Document, VectorStore, Embeddings


class ChromaVectorStore(VectorStore):
    def __init__(self, collection_name: Optional[str] = None, embeddings: Optional[Embeddings] = None) -> None:
        self.persist_dir = settings.chroma_persist_dir
        ensure_dir(self.persist_dir)
        self.collection_name = collection_name or settings.collection_name
        self.embeddings = embeddings
        # LangChain의 Chroma 래퍼 인스턴스 구성
        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,  # Embeddings Protocol은 LC OpenAIEmbeddings 호환
            persist_directory=self.persist_dir,
        )

    def add_texts(self, texts: Sequence[str], metadatas: Sequence[Dict[str, Any]]) -> None:
        # 대량 추가 시 OpenAI 임베딩 토큰 한도 초과를 피하기 위해 배치 처리
        batch_size = 64
        texts_list = list(texts)
        metas_list = list(metadatas)
        for i in range(0, len(texts_list), batch_size):
            batch_texts = texts_list[i : i + batch_size]
            batch_metas = metas_list[i : i + batch_size]
            if not batch_texts:
                continue
            self.store.add_texts(texts=batch_texts, metadatas=batch_metas)
        # persist는 LangChain Chroma가 필요 시 자동 처리하지만 명시적으로 호출
        try:
            self.store.persist()
        except Exception:
            pass

    def similarity_search(self, query: str, k: int) -> List[Document]:
        results = self.store.similarity_search(query, k=k)
        # 결과는 이미 LangChain Document이므로 그대로 반환
        return results


