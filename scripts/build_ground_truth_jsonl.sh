#!/usr/bin/env bash
set -euo pipefail

# 프로젝트 루트 및 경로
PROJECT_ROOT="/Users/geun1/programming/project/RAG/advanced_RAG"
QA_DIR_DEFAULT="$PROJECT_ROOT/AI-Hub-data-test/Statutory-QA-Data"
OUTPUT_DEFAULT="$PROJECT_ROOT/test_data_set.jsonl"

if [[ "${1:-}" == "--from-llm" ]]; then
  shift
  DOCS_DIR_DEFAULT="$PROJECT_ROOT/AI-Hub-data-test/Statutory-Source-Data"
  DOCS_DIR="${DOCS_DIR:-$DOCS_DIR_DEFAULT}"
  OUTPUT_JSONL="${OUTPUT_JSONL:-$OUTPUT_DEFAULT}"
  COUNT_PER_FILE="${COUNT_PER_FILE:-8}"
  TYPES_DEFAULT="simple,multihop,conditional,case,summary,cross,paraphrase"
  TYPES="${TYPES:-$TYPES_DEFAULT}"
  TYPE_COUNTS="${TYPE_COUNTS:-}"
  MODEL="${MODEL:-${CHAT_MODEL:-}}"
  MAX_CHARS_PER_DOC="${MAX_CHARS_PER_DOC:-12000}"
  MAX_CROSS_NEIGHBORS="${MAX_CROSS_NEIGHBORS:-1}"
  DRY_RUN_FLAG="${DRY_RUN:-}"

  cmd=(python3 "$PROJECT_ROOT/scripts/generate_test_dataset.py" \
    --docs-dir "$DOCS_DIR" \
    --output "$OUTPUT_JSONL" \
    --count-per-file "$COUNT_PER_FILE" \
    --types "$TYPES" \
    --max-chars-per-doc "$MAX_CHARS_PER_DOC" \
    --max-cross-neighbors "$MAX_CROSS_NEIGHBORS")

  if [[ -n "$TYPE_COUNTS" ]]; then
    cmd+=(--type-counts "$TYPE_COUNTS")
  fi
  if [[ -n "$MODEL" ]]; then
    cmd+=(--model "$MODEL")
  fi
  if [[ -n "$DRY_RUN_FLAG" ]]; then
    cmd+=(--dry-run)
  fi

  if [[ $# -gt 0 ]]; then
    cmd+=("$@")
  fi

  echo "[INFO] LLM 기반 데이터셋 생성 실행:" "${cmd[@]}"
  "${cmd[@]}"
  exit 0
fi

QA_DIR="${1:-$QA_DIR_DEFAULT}"
OUTPUT_JSONL="${2:-$OUTPUT_DEFAULT}"
MAX_ITEMS="${3:-150}"

if [[ ! -d "$QA_DIR" ]]; then
  echo "[ERROR] QA 디렉터리를 찾을 수 없습니다: $QA_DIR" >&2
  exit 1
fi

python3 - <<PY
import os, json, sys
from glob import iglob

qa_dir = os.environ.get('QA_DIR') or sys.argv[1] if len(sys.argv) > 1 else "$QA_DIR"
out_path = os.environ.get('OUTPUT_JSONL') or sys.argv[2] if len(sys.argv) > 2 else "$OUTPUT_JSONL"
max_items = int(os.environ.get('MAX_ITEMS') or (sys.argv[3] if len(sys.argv) > 3 else "$MAX_ITEMS"))

items = []
qa_paths = sorted(iglob(os.path.join(qa_dir, "**", "*.json"), recursive=True))
for p in qa_paths:
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)
        label = obj.get("label", {})
        info = obj.get("info", {})
        question = str(label.get("input", "")).strip()
        answer = str(label.get("output", "")).strip()
        law_id = str(info.get("lawId", "")).strip()
        sources = []
        if law_id:
            sources = [f"HJ_B_{law_id}.csv"]
        if not sources:
            base = os.path.basename(p)
            prefix = base.split("_QA_")[0] if "_QA_" in base else os.path.splitext(base)[0]
            if prefix:
                sources = [f"{prefix}.csv"]
        if question and answer:
            items.append({
                "question": question,
                "ground_truth_answer": answer,
                "ground_truth_sources": sources,
            })
    except Exception as e:
        print(f"[WARN] QA 파싱 실패: {p}: {e}")

# 제한 개수 적용
items = items[:max_items]

with open(out_path, "w", encoding="utf-8") as out:
    for it in items:
        out.write(json.dumps(it, ensure_ascii=False) + "\n")
print(f"[DONE] QA 데이터셋 생성: {out_path} (총 {len(items)}개 항목)")
PY


