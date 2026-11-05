
from openai import AsyncOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from app.core.config import settings
from app.graph.state import AgentState


class EmailAgent:

    def __init__(self):

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured")

        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY
        )

        self.model = settings.OPENAI_CHAT_MODEL

    async def generate_draft(self, query: str) -> str:

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional business email writing assistant. "
                        "Generate clear, concise and professional email drafts. "
                        "Never claim that an email has been sent. "
                        "Return only the email draft."
                    ),
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content or ""

    async def close(self):
        await self.client.close()


async def email_agent_node(state: AgentState) -> dict:

    query = state.get("query", "").strip()
    messages = state.get("messages", [])

    if not query:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                query = message.content
                break

    if not query:
        raise ValueError("No email drafting request found")

    agent = EmailAgent()

    try:
        draft = await agent.generate_draft(query)

        return {
            "messages": [
                AIMessage(content=draft)
            ],
            "email_draft": draft,
            "response": draft,
            "next_agent": "END",
            "error": "",
        }

    finally:
        await agent.close()
