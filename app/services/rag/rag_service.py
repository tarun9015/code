
from openai import AsyncOpenAI
from langsmith import traceable

from app.core.config import settings
from app.services.rag.prompt import SYSTEM_PROMPT


class RAGService:

    def __init__(self):

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured")

        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY
        )

        self.model = settings.OPENAI_CHAT_MODEL

    @traceable(
        name="llm_generation",
        run_type="llm",
    )
    async def generate_answer(
        self,
        question: str,
        retrieved_contexts: list[str],
    ) -> dict:
        """
        Generate an answer using context retrieved through MCP.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if not retrieved_contexts:
            return {
                "answer": (
                    "I could not find sufficient information "
                    "in the available documents."
                ),
                "usage": None,
                "model": self.model,
            }

        context = "\n\n".join(retrieved_contexts)

        system_prompt = SYSTEM_PROMPT.format(
            context=context,
            question=question,
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ],
            temperature=0,
        )

        return {
            "answer": response.choices[0].message.content or "",
            "usage": (
                response.usage.model_dump()
                if response.usage
                else None
            ),
            "model": self.model,
        }

    async def close(self) -> None:
        await self.client.close()
