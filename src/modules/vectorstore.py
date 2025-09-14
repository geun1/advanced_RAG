from typing import Any, Dict, List, Sequence, Optional
import chromadb

from ..config import settings
from .utils import ensure_dir
from .types import Document, VectorStore, Embeddings


class ChromaVectorStore(VectorStore):
    def __init__(self, collection_name: Optional[str] = None, embeddings: Optional[Embeddings] = None) -> None:
        self.persist_dir = settings.chroma_persist_dir
        ensure_dir(self.persist_dir)
        # Chroma 0.5.x에서는 PersistentClient 사용 및 별도 persist() 호출이 필요하지 않음
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name or settings.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embeddings = embeddings

    def add_texts(self, texts: Sequence[str], metadatas: Sequence[Dict[str, Any]]) -> None:
        ids = [f"doc-{i}-{m.get('chunk_id', i)}" for i, m in enumerate(metadatas)]
        if self.embeddings is not None:
            vecs = self.embeddings.embed_documents(texts)
            self.collection.add(documents=list(texts), embeddings=vecs, metadatas=list(metadatas), ids=ids)
        else:
            # 컬렉션에 embedding_function이 설정되어 있어야 함
            self.collection.add(documents=list(texts), metadatas=list(metadatas), ids=ids)
        # PersistentClient는 자동 영속화되므로 별도 persist 호출 불필요

    def similarity_search(self, query: str, k: int) -> List[Document]:
        if self.embeddings is not None:
            q = self.embeddings.embed_documents([query])[0]
            result = self.collection.query(query_embeddings=[q], n_results=k)
        else:
            result = self.collection.query(query_texts=[query], n_results=k)
        docs: List[Document] = []
        for i in range(len(result["ids"][0])):
            docs.append(
                Document(
                    page_content=result["documents"][0][i],
                    metadata=result["metadatas"][0][i],
                )
            )
        return docs


