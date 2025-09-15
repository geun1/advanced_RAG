#!/usr/bin/env bash
set -euo pipefail

# 사용법:
#   ./prune_qa_files.sh [QA_DIR] [MAX_PER=5] [--dry-run]
# 예:
#   ./prune_qa_files.sh                                   # 기본 경로, 5개 유지, 실제 삭제
#   ./prune_qa_files.sh /path/to/QA 3 --dry-run           # 미리 보기만(삭제 안함)

PROJECT_ROOT="/Users/geun1/programming/project/RAG/advanced_RAG"
QA_DIR_DEFAULT="$PROJECT_ROOT/AI-Hub-data-test/Statutory-QA-Data"

QA_DIR="${1:-$QA_DIR_DEFAULT}"
MAX_PER="${2:-3}"
FLAG="${3:-}"

if [[ ! -d "$QA_DIR" ]]; then
  echo "[ERROR] QA 디렉터리를 찾을 수 없습니다: $QA_DIR" >&2
  exit 1
fi

DRY_RUN=0
if [[ "$FLAG" == "--dry-run" ]]; then
  DRY_RUN=1
fi

export QA_DIR MAX_PER DRY_RUN
python3 - <<'PY'
import os, re, sys
from glob import iglob

qa_dir = os.environ["QA_DIR"]
max_per = int(os.environ.get("MAX_PER", 3))
dry_run = os.environ.get("DRY_RUN", "0") == "1"

pattern = re.compile(r"^HJ_B_(\d+)_QA_(\d+)\.json$")

# 파일 수집 (하위 폴더 포함)
paths = sorted(iglob(os.path.join(qa_dir, "**", "*.json"), recursive=True))
groups = {}
for p in paths:
    base = os.path.basename(p)
    m = pattern.match(base)
    if not m:
        continue
    key = m.group(1)  # 법령 ID 숫자 부분
    num = int(m.group(2))  # QA 번호
    groups.setdefault(key, []).append((num, p))

to_delete = []
to_keep = []
for key, items in groups.items():
    # 번호 오름차순으로 정렬 후 앞에서 max_per개 유지
    items.sort(key=lambda x: x[0])
    keep = items[:max_per]
    drop = items[max_per:]
    to_keep.extend(keep)
    to_delete.extend(drop)

print(f"[INFO] 그룹 수: {len(groups)} / 유지 {len(to_keep)} / 삭제 {len(to_delete)}")
if to_delete:
    print("[INFO] 삭제 대상 목록 (일부):")
    for _, p in to_delete[:20]:
        print("  -", p)
    if len(to_delete) > 20:
        print(f"  ... (총 {len(to_delete)}개)")

if dry_run:
    print("[DRY-RUN] 실제 삭제는 수행하지 않습니다.")
    sys.exit(0)

for _, p in to_delete:
    try:
        os.remove(p)
    except Exception as e:
        print(f"[WARN] 삭제 실패: {p}: {e}")

print(f"[DONE] 삭제 완료: {len(to_delete)}개 파일 삭제, 각 그룹당 최대 {max_per}개 유지")
PY


