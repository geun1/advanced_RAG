import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "docs")

    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Rerank settings
    rerank_enabled: bool = os.getenv("RERANK_ENABLED", "false").lower() in {"1", "true", "yes", "y"}
    rerank_provider: str = os.getenv("RERANK_PROVIDER", "cohere")
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "4"))

    # Cohere Rerank
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    cohere_rerank_model: str = os.getenv("COHERE_RERANK_MODEL", "rerank-multilingual-v3.0")

    # Elasticsearch (Sparse) settings
    elastic_host: str = os.getenv("ELASTIC_HOST", "http://localhost:9200")
    elastic_index: str = os.getenv("ELASTIC_INDEX", "docs")
    elastic_username: str = os.getenv("ELASTIC_USERNAME", "")
    elastic_password: str = os.getenv("ELASTIC_PASSWORD", "")
    elastic_verify_certs: bool = os.getenv("ELASTIC_VERIFY_CERTS", "true").lower() in {"1", "true", "yes", "y"}


settings = Settings()


