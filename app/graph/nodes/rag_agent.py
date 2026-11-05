import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

from app.graph.state import AgentState
from app.services.rag.rag_service import RAGService
from app.services.rag.query_rewriter import QueryRewriter


async def rag_agent_node(state: AgentState) -> dict:
    """
    LangGraph RAG Agent.

    Workflow:
    1. Rewrite the user's query using conversation history.
    2. Retrieve documents through an independent MCP server.
    3. Generate an answer using the retrieved document context.
    """

    messages = state.get("messages", [])
    query = state.get("query", "").strip()

    if not query:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                query = message.content
                break

    if not query:
        raise ValueError("No user query found in agent state")

    # Build conversation history, excluding the latest question.
    conversation_history = []

    for message in messages[:-1]:
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

    # Step 1: Rewrite the question.
    query_rewriter = QueryRewriter()

    try:
        rewritten_query = await query_rewriter.rewrite(
            question=query,
            conversation_history=conversation_history,
        )
    finally:
        await query_rewriter.close()

    print("Original Query:", query)
    print("Rewritten Query:", rewritten_query)

    # Step 2: Connect to the independent MCP server.
    project_path = Path(__file__).resolve().parents[3]

    transport = PythonStdioTransport(
        script_path=str(project_path / "app/mcp/server.py"),
        cwd=str(project_path),
        env={
            **os.environ,
            "PYTHONPATH": str(project_path),
        },
    )

    async with Client(transport) as client:
        mcp_result = await client.call_tool(
            "search_company_documents",
            {
                "query": rewritten_query,
                "top_k": 5,
            },
        )

    if mcp_result.is_error:
        raise RuntimeError("MCP document retrieval failed")

    retrieval_data = mcp_result.data or {}
    retrieved_chunks = retrieval_data.get("results", [])

    # Step 3: Extract context and source metadata.
    retrieved_contexts = [
        chunk["content"]
        for chunk in retrieved_chunks
        if chunk.get("content")
    ]

    sources = [
        {
            key: value
            for key, value in chunk.items()
            if key != "content"
        }
        for chunk in retrieved_chunks
    ]

    # Step 4: Generate the answer.
    rag_service = RAGService()

    try:
        result = await rag_service.generate_answer(
            question=rewritten_query,
            retrieved_contexts=retrieved_contexts,
        )
    finally:
        await rag_service.close()

    answer = result["answer"]

    # Step 5: Return the result to LangGraph.
    return {
        "messages": [
            AIMessage(content=answer)
        ],
        "rewritten_query": rewritten_query,
        "retrieved_chunks": retrieved_chunks,
        "sources": sources,
        "response": answer,
        "next_agent": "END",
        "error": "",
    }