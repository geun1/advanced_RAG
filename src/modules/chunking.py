from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings
from .types import TextSplitter


class DefaultTextSplitter(TextSplitter):
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split_text(text)


