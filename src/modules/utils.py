import hashlib
import os
from pathlib import Path
from typing import List


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def list_files_recursively(root: str, exts: List[str]) -> List[str]:
    results: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not exts or any(name.lower().endswith(ext) for ext in exts):
                results.append(os.path.join(dirpath, name))
    return sorted(results)


