
import asyncio
from uuid import uuid4

from langchain_core.messages import HumanMessage

from app.memory.checkpointer import get_checkpointer
from app.graph.graph_builder import build_graph


async def main():

    thread_id = f"enterprise-rag-test-{uuid4()}"

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    initial_state = {
        "messages": [
            HumanMessage(
                content="How many annual leaves are employees entitled to?"
            )
        ],
        "query": "How many annual leaves are employees entitled to?",
        "next_agent": "",
        "retrieved_chunks": [],
        "sources": [],
        "web_results": [],
        "email_draft": "",
        "response": "",
        "error": "",
    }

    print("\nENTERPRISE AGENTIC RAG EXECUTION")
    print("=" * 55)

    async with get_checkpointer() as checkpointer:

        graph = build_graph(
            checkpointer=checkpointer
        )

        # Execute the complete graph.
        result = await graph.ainvoke(
            initial_state,
            config=config,
        )

        print("\nSelected Agent:")
        print(result["next_agent"])

        print("\nFinal Response:")
        print(result["response"])

        print("\nRetrieved Sources:")
        print(result["sources"])

        # Verify checkpoint persistence.
        saved_state = await graph.aget_state(config)

        if not saved_state.values:
            raise RuntimeError("Graph checkpoint was not saved")

        print("\nCheckpoint Persistence: SUCCESS")
        print("Thread ID:", thread_id)

        print("\nComplete graph execution passed!")


if __name__ == "__main__":
    asyncio.run(main())
