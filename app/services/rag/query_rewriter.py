
from openai import AsyncOpenAI
from langsmith import traceable

from app.core.config import settings


class QueryRewriter:

    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured")

        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY
        )
        self.model = settings.OPENAI_CHAT_MODEL

    @traceable(name="conversation_query_rewriting", run_type="llm")
    async def rewrite(
        self,
        question: str,
        conversation_history: list[dict],
    ) -> str:

        if not conversation_history:
            return question

        history = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in conversation_history[-10:]
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a query rewriting assistant for an enterprise RAG system.

Your task is to rewrite the user's latest question into a
standalone question using the conversation history.

Rules:
1. Resolve references such as "it", "that", "they", and "their"
   using the conversation history.
2. Preserve the original intent of the latest question.
3. Do not answer the question.
4. Do not add facts that are not present in the conversation.
5. If the latest question is already standalone, return it unchanged.
6. Return only the rewritten question.
"""
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{history}\n\n"
                        f"Latest question:\n{question}"
                    ),
                },
            ],
            temperature=0,
        )

        rewritten_query = response.choices[0].message.content

        return (rewritten_query or question).strip()

    async def close(self):
        await self.client.close()
