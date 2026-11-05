
import asyncio

from app.graph.nodes.supervisor import SupervisorAgent


async def main():

    test_queries = [
        {
            "query": "How many annual leaves are employees entitled to?",
            "expected": "rag",
        },
        {
            "query": "Find the latest developments in Generative AI.",
            "expected": "web_search",
        },
        {
            "query": "Draft an email requesting approval for vendor invoices.",
            "expected": "email",
        },
        {
            "query": "Explain the difference between Python lists and tuples.",
            "expected": "conversational",
        },
    ]

    supervisor = SupervisorAgent()

    try:
        print("\nSUPERVISOR ROUTING TEST")
        print("=" * 60)

        for test in test_queries:

            decision = await supervisor.route(test["query"])

            passed = decision.next_agent == test["expected"]

            print(f"\nQuery: {test['query']}")
            print(f"Expected: {test['expected']}")
            print(f"Selected: {decision.next_agent}")
            print(f"Reasoning: {decision.reasoning}")
            print(f"Result: {'PASS' if passed else 'FAIL'}")

        print("\nSupervisor routing test completed!")

    finally:
        await supervisor.close()


if __name__ == "__main__":
    asyncio.run(main())

