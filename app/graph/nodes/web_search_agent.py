
import asyncio

from ddgs import DDGS
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.state import AgentState


class WebSearchAgent:

    def __init__(self):
        self.max_results = 5

    async def search(self, query: str) -> list[dict]:
        """
        Execute DuckDuckGo search asynchronously.
        The synchronous search operation runs in a worker thread.
        """

        def perform_search():
            return DDGS().text(
                query,
                max_results=self.max_results,
            )

        results = await asyncio.to_thread(perform_search)

        return [
            {
                "title": item.get("title", "N/A"),
                "snippet": item.get("body", "N/A"),
                "link": item.get("href", "N/A"),
            }
            for item in results
        ]


async def web_search_agent_node(state: AgentState) -> dict:

    query = state.get("query", "").strip()
    messages = state.get("messages", [])

    if not query:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                query = message.content
                break

    if not query:
        raise ValueError("No search query found in agent state")

    agent = WebSearchAgent()

    try:
        results = await agent.search(query)

        if results:
            answer = "\n\n".join(
                [
                    (
                        f"Title: {item['title']}\n"
                        f"Snippet: {item['snippet']}\n"
                        f"Link: {item['link']}"
                    )
                    for item in results
                ]
            )
        else:
            answer = "No relevant web search results were found."

        return {
            "messages": [AIMessage(content=answer)],
            "web_results": results,
            "response": answer,
            "next_agent": "END",
            "error": "",
        }

    except Exception as exc:
        return {
            "error": f"Web search failed: {str(exc)}",
            "response": "I could not complete the web search.",
            "next_agent": "END",
        }
