from typing import Dict, List, Optional
from pathlib import Path
from pypdf import PdfReader

from .types import Document


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf_file(path: str) -> str:
    reader = PdfReader(path)
    texts: List[str] = []
    for page in reader.pages:
        content = page.extract_text() or ""
        texts.append(content)
    return "\n".join(texts)


def load_file_to_document(path: str, extra_metadata: Optional[Dict[str, str]] = None) -> Document:
    ext = Path(path).suffix.lower()
    if ext in [".txt", ".md", ".py", ".sql", ".csv", ".json", ".yaml", ".yml"]:
        text = load_text_file(path)
    elif ext in [".pdf"]:
        text = load_pdf_file(path)
    else:
        text = load_text_file(path)

    metadata: Dict[str, str] = {"source": str(path)}
    if extra_metadata:
        metadata.update(extra_metadata)
    return Document(page_content=text, metadata=metadata)


