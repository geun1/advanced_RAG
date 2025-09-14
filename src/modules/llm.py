from typing import Sequence, Optional
from openai import OpenAI

from ..config import settings
from .types import ChatLLM, Document


SYSTEM_PROMPT = (
    "You are a helpful RAG assistant. Use the provided context to answer the question. "
    "If the answer is not in context, say you don't know briefly in Korean."
)


class OpenAIChatLLM(ChatLLM):
    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or settings.chat_model
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, question: str, context_docs: Sequence[Document], max_tokens: Optional[int] = None) -> str:
        context_text = "\n\n".join(
            [f"[소스: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in context_docs]
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"질문:\n{question}\n\n컨텍스트:\n{context_text}"},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens or settings.max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


