
import asyncio

from fastmcp import Client
from app.mcp.server import mcp


async def main():

    async with Client(mcp) as client:

        # List registered MCP tools.
        tools = await client.list_tools()

        print("\nRegistered MCP Tools:")

        for tool in tools:
            print(f"- {tool.name}")

        # Test document search through MCP.
        result = await client.call_tool(
            "search_company_documents",
            {
                "query": "How many annual leaves are employees entitled to?",
                "top_k": 5,
            },
        )

        print("\nMCP Document Search Results:")
        print(result.data)

        # Validate search results.
        assert not result.is_error, "MCP tool execution failed"

        assert result.data["total_results"] > 0, (
            "No document search results returned"
        )

        first_result = result.data["results"][0]

        assert "reranker_score" in first_result, (
            "Reranker score missing"
        )

        print("\nMCP_DOCUMENT_SEARCH_TEST_PASSED")


if __name__ == "__main__":
    asyncio.run(main())
