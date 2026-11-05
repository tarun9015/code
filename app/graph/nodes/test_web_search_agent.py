
import asyncio
from pprint import pprint

from langchain_core.messages import HumanMessage

from app.graph.nodes.web_search_agent import web_search_agent_node


async def main():

    state = {
        "messages": [
            HumanMessage(
                content="What are the latest developments in generative AI?"
            )
        ],
        "query": "Latest developments in generative AI",
        "next_agent": "web_search",
        "retrieved_chunks": [],
        "sources": [],
        "web_results": [],
        "email_draft": "",
        "response": "",
        "error": "",
    }

    result = await web_search_agent_node(state)

    print("\nWEB SEARCH AGENT RESULT")
    print("=" * 50)

    print("\nResponse:")
    print(result["response"])

    print("\nSearch Results:")
    pprint(result["web_results"])

    print("\nWeb Search Agent test completed!")


if __name__ == "__main__":
    asyncio.run(main())
