#!/usr/bin/env bash
set -euo pipefail

# 프로젝트 루트와 데이터 디렉터리(절대 경로)
PROJECT_ROOT="/Users/geun1/programming/project/RAG/advanced_RAG"
DATA_DIR="$PROJECT_ROOT/AI-Hub-data-test/Statutory-Source-Data"

# .env 자동 로드(있으면)
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$PROJECT_ROOT/.env"
  set +a
fi

# 필수 키 확인
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY 환경변수가 필요합니다(.env 설정 또는 export)." >&2
  echo "예: export OPENAI_API_KEY=sk-..." >&2
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[ERROR] 데이터 디렉터리를 찾을 수 없습니다: $DATA_DIR" >&2
  exit 1
fi


export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

echo "[INFO] CSV 수집 중: $DATA_DIR"

python3 - <<'PY'
import os
from glob import iglob

from src.modules.loaders import load_file_to_document
from src.modules.chunking import DefaultTextSplitter
from src.modules.embeddings import OpenAIEmbeddings
from src.modules.vectorstore import ChromaVectorStore
from src.modules.retriever import SimpleRetriever
from src.modules.llm import OpenAIChatLLM
from src.modules.pipeline import RAGPipeline

PROJECT_ROOT = "/Users/geun1/programming/project/RAG/advanced_RAG"
DATA_DIR = os.path.join(PROJECT_ROOT, "AI-Hub-data-test/Statutory-Source-Data")

# 컴포넌트 초기화
splitter = DefaultTextSplitter()
embeddings = OpenAIEmbeddings()
store = ChromaVectorStore(embeddings=embeddings)
retriever = SimpleRetriever(store)
llm = OpenAIChatLLM()
pipeline = RAGPipeline(splitter, embeddings, store, retriever, llm)

# CSV 파일 전부 로드 → Document 리스트 생성
csv_paths = sorted(iglob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True))
if not csv_paths:
    raise SystemExit("[INFO] 색인할 CSV가 없습니다.")

docs = []
for p in csv_paths:
    meta = {"dataset": "AI-Hub", "basename": os.path.basename(p)}
    docs.append(load_file_to_document(p, extra_metadata=meta))

print(f"[INFO] 총 {len(docs)}개 CSV를 문서로 변환했습니다. 청킹/임베딩/저장을 시작합니다...")

stats = pipeline.index_documents(docs)
print("[DONE] 색인 완료:", stats)
PY


