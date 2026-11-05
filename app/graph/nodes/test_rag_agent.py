
import asyncio
from pprint import pprint

from langchain_core.messages import HumanMessage

from app.graph.nodes.rag_agent import rag_agent_node


async def main():

    initial_state = {
        "messages": [
            HumanMessage(
                content="How many annual leaves are employees entitled to?"
            )
        ],
        "query": "How many annual leaves are employees entitled to?",
        "next_agent": "rag",
        "retrieved_chunks": [],
        "sources": [],
        "web_results": [],
        "email_draft": "",
        "response": "",
        "error": "",
    }

    result = await rag_agent_node(initial_state)

    print("\nRAG AGENT NODE RESULT")
    print("=" * 50)

    print("\nResponse:")
    print(result["response"])

    print("\nSources:")
    pprint(result["sources"])

    print("\nRetrieved Chunks:")
    pprint(result["retrieved_chunks"])

    print("\nRAG Agent node test passed!")


if __name__ == "__main__":
    asyncio.run(main())
