
import asyncio

from langchain_core.messages import HumanMessage

from app.graph.nodes.email_agent import email_agent_node


async def main():

    state = {
        "messages": [
            HumanMessage(
                content=(
                    "Draft an email to the Finance Manager requesting "
                    "approval for pending vendor invoices."
                )
            )
        ],
        "query": (
            "Draft an email to the Finance Manager requesting "
            "approval for pending vendor invoices."
        ),
        "next_agent": "email",
        "retrieved_chunks": [],
        "sources": [],
        "web_results": [],
        "email_draft": "",
        "response": "",
        "error": "",
    }

    result = await email_agent_node(state)

    print("\nEMAIL AGENT RESULT")
    print("=" * 50)

    print("\nGenerated Email Draft:")
    print(result["email_draft"])

    print("\nEmail Agent test passed!")
    print("No email was sent.")


if __name__ == "__main__":
    asyncio.run(main())
