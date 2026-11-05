
import asyncio

from app.memory.checkpointer import get_checkpointer
from app.graph.graph_builder import build_graph


async def main():

    print("\nLANGGRAPH COMPILATION TEST")
    print("=" * 50)

    async with get_checkpointer() as checkpointer:

        graph = build_graph(
            checkpointer=checkpointer
        )

        print("Graph compilation: SUCCESS")
        print("PostgreSQL checkpointer: ENABLED")

        print("\nRegistered graph nodes:")
        print(list(graph.nodes.keys()))

        expected_nodes = {
            "supervisor",
            "rag",
            "web_search",
            "email",
            "conversational",
        }

        actual_nodes = set(graph.nodes.keys())

        if not expected_nodes.issubset(actual_nodes):
            raise RuntimeError(
                "One or more expected graph nodes are missing"
            )

        print("\nAll expected nodes registered successfully.")
        print("Graph compilation test passed!")


if __name__ == "__main__":
    asyncio.run(main())
