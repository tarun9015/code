
from openai import AsyncOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from app.core.config import settings
from app.graph.state import AgentState


class ConversationalAgent:

    def __init__(self):

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured")

        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY
        )

        self.model = settings.OPENAI_CHAT_MODEL

    async def run(self, query: str, messages: list) -> str:

        conversation_history = []

        for message in messages[-10:]:

            if isinstance(message, HumanMessage):
                conversation_history.append({
                    "role": "user",
                    "content": message.content,
                })

            elif isinstance(message, AIMessage):
                conversation_history.append({
                    "role": "assistant",
                    "content": message.content,
                })

        if not conversation_history:
            conversation_history.append({
                "role": "user",
                "content": query,
            })

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful enterprise conversational assistant. "
                        "Answer clearly and professionally."
                    ),
                },
                *conversation_history,
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content or ""

    async def close(self):
        await self.client.close()


async def conversational_agent_node(state: AgentState) -> dict:

    query = state.get("query", "").strip()
    messages = state.get("messages", [])

    if not query:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                query = message.content
                break

    if not query:
        raise ValueError("No user query found in agent state")

    agent = ConversationalAgent()

    try:
        answer = await agent.run(
            query=query,
            messages=messages,
        )

        return {
            "messages": [
                AIMessage(content=answer)
            ],
            "response": answer,
            "next_agent": "END",
            "error": "",
        }

    finally:
        await agent.close()
