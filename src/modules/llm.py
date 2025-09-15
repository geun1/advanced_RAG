from typing import Sequence, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..config import settings
from .types import ChatLLM, Document


SYSTEM_PROMPT = (
    "You are a helpful RAG assistant. Use the provided context to answer the question. "
    "Write your answers so that you can fully understand them without having to look at the reference document again. "
    "Be sure to use specific names instead of ambiguous pronouns (e.g., 'this,' 'appropriate'). "
    "If the answer is not in context, say you don't know briefly in Korean."
)


class OpenAIChatLLM(ChatLLM):
    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or settings.chat_model
        self.llm = ChatOpenAI(model=self.model, api_key=settings.openai_api_key, temperature=0.2)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT + "\n문맥:\n{context}"),
            ("human", "질문: {question}"),
        ])
        self.parser = StrOutputParser()

    def generate(self, question: str, context_docs: Sequence[Document], max_tokens: Optional[int] = None) -> str:
        context_text = "\n\n".join(
            [f"[소스: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in context_docs]
        )
        chain = self.prompt | self.llm | self.parser
        return chain.invoke({"context": context_text, "question": question})


