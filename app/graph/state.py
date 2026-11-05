
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    next_agent: str
    selected_agent: str
    rewritten_query: str
    retrieved_chunks: list[dict]
    sources: list[dict]
    web_results: list[dict]
    email_draft: str
    response: str
    error: str
