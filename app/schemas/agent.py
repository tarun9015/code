
from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    """Request schema for the Agent Chat API."""

    query: str = Field(
        ...,
        min_length=1,
        description="User's message or question",
    )

    thread_id: str | None = Field(
        default=None,
        description="Existing conversation ID. "
                    "Leave empty to start a new conversation.",
    )


class AgentChatResponse(BaseModel):
    """Response schema for the Agent Chat API."""

    thread_id: str

    selected_agent: str

    rewritten_query: str | None = None

    response: str

    sources: list[dict] = Field(default_factory=list)

    retrieved_chunks: list[dict] = Field(default_factory=list)

    web_results: list[dict] = Field(default_factory=list)

    email_draft: str = ""
