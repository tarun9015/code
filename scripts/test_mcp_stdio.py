import asyncio

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport


async def main():
    project_path = "/Users/tarunmac/projects/enterprise-agentic-rag-v2"

    transport = PythonStdioTransport(
        script_path="app/mcp/server.py",
        cwd=project_path,
        env={
            "PYTHONPATH": project_path,
        },
    )

    async with Client(transport) as client:

        print("\nMCP STDIO SERVER TEST")
        print("=" * 50)

        tools = await client.list_tools()

        print("\nRegistered MCP Tools:")

        for tool in tools:
            print(f"- {tool.name}")

        tool_names = {tool.name for tool in tools}

        assert "search_company_documents" in tool_names
        assert "health_check" in tool_names

        health_result = await client.call_tool(
            "health_check",
            {},
        )

        assert not health_result.is_error

        print("\nHealth Check:")
        print(health_result.data)

        result = await client.call_tool(
            "search_company_documents",
            {
                "query": "How many annual leaves are employees entitled to?",
                "top_k": 5,
            },
        )

        assert not result.is_error

        data = result.data

        print("\nDocument Search Result:")
        print(data)

        assert data["total_results"] > 0

        first_result = data["results"][0]

        assert "content" in first_result
        assert "reranker_score" in first_result

        print("\nMCP STDIO INTEGRATION TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())