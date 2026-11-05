
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from app.graph.graph_builder import build_graph
from app.memory.checkpointer import get_checkpointer
from app.schemas.agent import (
    AgentChatRequest,
    AgentChatResponse,
)


router = APIRouter()


@router.post(
    "/chat",
    response_model=AgentChatResponse,
)
async def agent_chat(
    request: AgentChatRequest,
):

    thread_id = request.thread_id or str(uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    initial_state = {
        "messages": [
            HumanMessage(content=request.query)
        ],
        "query": request.query,
        "next_agent": "",
        "selected_agent": "",
        "rewritten_query": "",
        "retrieved_chunks": [],
        "sources": [],
        "web_results": [],
        "email_draft": "",
        "response": "",
        "error": "",
    }

    try:
        async with get_checkpointer() as checkpointer:

            graph = build_graph(
                checkpointer=checkpointer
            )

            result = await graph.ainvoke(
                initial_state,
                config=config,
            )

            return AgentChatResponse(
                thread_id=thread_id,
                selected_agent=result.get(
                    "selected_agent",
                    "unknown",
                ),
                rewritten_query=result.get(
                    "rewritten_query"
                ),
                response=result.get("response", ""),
                sources=result.get("sources", []),
                retrieved_chunks=result.get(
                    "retrieved_chunks", []
                ),
                web_results=result.get("web_results", []),
                email_draft=result.get("email_draft", ""),
            )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Agent execution failed",
        ) from exc
