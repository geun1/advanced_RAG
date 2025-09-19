from typing import Any, Dict, List, Sequence, Optional
import json
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from urllib.parse import urlparse
import time

from ..config import settings
from .utils import ensure_dir
from .types import VectorStore, Embeddings

try:
    from elasticsearch import Elasticsearch, helpers  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Elasticsearch = None  # type: ignore
    helpers = None  # type: ignore


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
        metas_list = [self._normalize_metadata(m) for m in list(metadatas)]
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

    def _normalize_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        def norm_val(v: Any) -> Any:
            # Chroma 허용: str, int, float, bool, None
            if v is None or isinstance(v, (str, int, float, bool)):
                return v
            # 리스트/튜플/세트는 콤마 문자열로 변환
            if isinstance(v, (list, tuple, set)):
                try:
                    return ",".join(str(x) for x in v)
                except Exception:
                    return json.dumps(list(v), ensure_ascii=False)
            # 딕셔너리는 JSON 문자열로 변환
            if isinstance(v, dict):
                try:
                    return json.dumps(v, ensure_ascii=False)
                except Exception:
                    return str(v)
            # 기타는 문자열로 강제 변환
            return str(v)

        return {k: norm_val(v) for k, v in meta.items()}



class ElasticVectorStore(VectorStore):
    """Elasticsearch를 이용한 Sparse(키워드/BM25) 기반 VectorStore 호환 래퍼.

    - add_texts: 문서와 메타데이터를 ES 인덱스에 색인
    - similarity_search: BM25 기반 full-text 검색으로 상위 k개 반환
    """

    def __init__(self, index_name: Optional[str] = None) -> None:
        self.index = index_name or settings.elastic_index
        self.available = False
        self.client: Optional[Elasticsearch] = None  # type: ignore
        self.last_error: Optional[str] = None

        if Elasticsearch is None:
            return

        # 호스트 파싱(문자열 URL → dict(host, port, scheme))로 호환성 개선
        parsed = None
        try:
            parsed = urlparse(settings.elastic_host)
        except Exception:
            parsed = None
        hosts_cfg: List[Any] = []
        if parsed and parsed.scheme and parsed.hostname and parsed.port:
            hosts_cfg = [{"host": parsed.hostname, "port": parsed.port, "scheme": parsed.scheme}]
        else:
            hosts_cfg = [settings.elastic_host]

        es_kwargs: Dict[str, Any] = {"hosts": hosts_cfg}
        # ES 8 클러스터 호환 헤더 강제 (elasticsearch-py v9가 9 헤더를 보내 400 발생 방지)
        es_kwargs["headers"] = {
            "accept": "application/vnd.elasticsearch+json; compatible-with=8",
            "content-type": "application/vnd.elasticsearch+json; compatible-with=8",
        }
        if settings.elastic_username and settings.elastic_password:
            es_kwargs["basic_auth"] = (settings.elastic_username, settings.elastic_password)
        # HTTPS에서만 인증서 검증 플래그 적용
        try:
            scheme = (parsed.scheme.lower() if parsed else urlparse(settings.elastic_host).scheme.lower())
        except Exception:
            scheme = "http"
        if scheme == "https":
            es_kwargs["verify_certs"] = bool(settings.elastic_verify_certs)

        try:
            self.client = Elasticsearch(**es_kwargs)  # type: ignore
            # 초기 부팅 대기 및 재시도 (최대 10회, 0.5s 간격). info() 우선 시도
            ok = False
            for _ in range(10):
                try:
                    # info()가 성공하면 클러스터 정상
                    info = self.client.info()  # type: ignore
                    if isinstance(info, dict) and info.get("version"):
                        ok = True
                        break
                except Exception as e:
                    self.last_error = f"info_failed_exception:{type(e).__name__}:{e}"
                # ping()도 병행 시도
                try:
                    if self.client.ping(request_timeout=5):  # type: ignore
                        ok = True
                        break
                except Exception as e:
                    self.last_error = f"ping_failed_exception:{type(e).__name__}:{e}"
                time.sleep(0.5)
            if not ok:
                self.available = False
                if self.last_error is None:
                    self.last_error = "ping_failed"
                return
            self._ensure_index()
            self.available = True
        except Exception as e:
            self.available = False
            self.client = None
            self.last_error = f"init_exception:{type(e).__name__}"

    def _ensure_index(self) -> None:
        if not self.client:
            return
        try:
            exists = self.client.indices.exists(index=self.index)  # type: ignore
            if not exists:  # type: ignore
                # 간단한 매핑: content=text, metadata=object
                body = {
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "metadata": {"type": "object", "enabled": True},
                        }
                    }
                }
                self.client.indices.create(index=self.index, **body)  # type: ignore
        except Exception as e:
            # 인덱스 존재/경합 등은 무시하되 에러 저장
            self.last_error = f"ensure_index_exception:{type(e).__name__}"

    def add_texts(self, texts: Sequence[str], metadatas: Sequence[Dict[str, Any]]) -> None:
        if not self.available or not self.client or not helpers:
            return
        actions = []
        metas_list = [self._normalize_metadata(m) for m in list(metadatas)]
        for text, meta in zip(texts, metas_list):
            actions.append({
                "_op_type": "index",
                "_index": self.index,
                "content": text,
                "metadata": meta,
            })
        if not actions:
            return
        try:
            helpers.bulk(self.client, actions)  # type: ignore
        except Exception as e:
            self.last_error = f"bulk_exception:{type(e).__name__}"

    def similarity_search(self, query: str, k: int) -> List[Document]:
        if not self.available or not self.client:
            return []
        hits: List[Dict[str, Any]] = []

        # 1) 엄격 검색(AND)
        body_and = {
            "query": {
                "match": {
                    "content": {
                        "query": query,
                        "operator": "and",
                    }
                }
            },
            "size": k,
        }
        try:
            res = self.client.search(index=self.index, body=body_and)  # type: ignore
            hits = res.get("hits", {}).get("hits", [])  # type: ignore
        except Exception as e:
            self.last_error = f"search_exception_and:{type(e).__name__}:{e}"
            hits = []

        # 2) 저엄격 검색(OR) - 결과 없을 때 재시도
        if not hits:
            body_or = {
                "query": {
                    "match": {
                        "content": {
                            "query": query,
                            "operator": "or",
                        }
                    }
                },
                "size": k,
            }
            try:
                res = self.client.search(index=self.index, body=body_or)  # type: ignore
                hits = res.get("hits", {}).get("hits", [])  # type: ignore
            except Exception as e:
                self.last_error = f"search_exception_or:{type(e).__name__}:{e}"
                hits = []

        # 3) simple_query_string 백업
        if not hits:
            body_sqs = {
                "query": {
                    "simple_query_string": {
                        "query": query,
                        "fields": ["content"],
                        "default_operator": "or",
                    }
                },
                "size": k,
            }
            try:
                res = self.client.search(index=self.index, body=body_sqs)  # type: ignore
                hits = res.get("hits", {}).get("hits", [])  # type: ignore
            except Exception as e:
                self.last_error = f"search_exception_sqs:{type(e).__name__}:{e}"
                hits = []

        docs: List[Document] = []
        for h in hits:
            src = h.get("_source", {})
            content = src.get("content", "")
            metadata = src.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"raw_metadata": str(metadata)}
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def _normalize_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        def norm_val(v: Any) -> Any:
            if v is None or isinstance(v, (str, int, float, bool)):
                return v
            if isinstance(v, (list, tuple, set)):
                try:
                    return ",".join(str(x) for x in v)
                except Exception:
                    return json.dumps(list(v), ensure_ascii=False)
            if isinstance(v, dict):
                try:
                    return v  # ES object로 저장
                except Exception:
                    return {"value": str(v)}
            return str(v)
        return {k: norm_val(v) for k, v in meta.items()}


class CompositeVectorStore(VectorStore):
    """여러 VectorStore에 동시에 색인(add_texts)을 전파하는 복합 스토어.

    검색은 사용처(리트리버)에서 개별 스토어를 앙상블하기 때문에 여기서는 미사용.
    similarity_search는 첫 번째 스토어 결과를 반환해 하위호환만 유지.
    """

    def __init__(self, stores: Sequence[VectorStore]) -> None:
        self.stores = list(stores)

    def add_texts(self, texts: Sequence[str], metadatas: Sequence[Dict[str, Any]]) -> None:
        for s in self.stores:
            try:
                s.add_texts(texts, metadatas)
            except Exception:
                # 개별 스토어 실패는 무시하고 진행
                pass

    def similarity_search(self, query: str, k: int) -> List[Document]:
        if not self.stores:
            return []
        try:
            return self.stores[0].similarity_search(query, k)
        except Exception:
            return []

