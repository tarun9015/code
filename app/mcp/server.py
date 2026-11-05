
from uuid import UUID

from fastmcp import FastMCP

from app.db.session import AsyncSessionLocal
from app.repositories.document_repository import DocumentRepository
from app.services.retrieval.retrieval_service import RetrievalService


mcp = FastMCP(
    name="Enterprise Agentic RAG MCP Server"
)


@mcp.tool
def health_check() -> dict:
    """Check whether the MCP server is running."""

    return {
        "status": "healthy",
        "server": "Enterprise Agentic RAG MCP Server",
    }


@mcp.tool
def get_project_info() -> dict:
    """Get basic information about the Enterprise Agentic RAG project."""

    return {
        "project": "Enterprise Agentic RAG",
        "framework": "FastAPI",
        "orchestration": "LangGraph",
        "database": "PostgreSQL with pgvector",
        "protocol": "Model Context Protocol",
    }


@mcp.tool
async def list_documents(
    source: str | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """List available documents from the company document registry."""

    async with AsyncSessionLocal() as session:

        repository = DocumentRepository(session)

        documents = await repository.list_documents(
            source=source,
            status=status,
            limit=limit,
            offset=offset,
        )

        return {
            "total_returned": len(documents),
            "documents": [
                {
                    "id": str(document.id),
                    "source": document.source,
                    "source_document_id": document.source_document_id,
                    "filename": document.filename,
                    "current_version": document.current_version,
                    "status": document.status,
                    "created_at": (
                        document.created_at.isoformat()
                        if document.created_at
                        else None
                    ),
                    "updated_at": (
                        document.updated_at.isoformat()
                        if document.updated_at
                        else None
                    ),
                }
                for document in documents
            ],
        }


@mcp.tool
async def get_document_details(
    document_id: str,
) -> dict:
    """Retrieve metadata for a specific document using its UUID."""

    try:
        parsed_document_id = UUID(document_id)
    except ValueError:
        return {
            "error": "Invalid document UUID format"
        }

    async with AsyncSessionLocal() as session:

        repository = DocumentRepository(session)

        document = await repository.get_by_id(
            parsed_document_id
        )

        if document is None:
            return {
                "error": "Document not found",
                "document_id": document_id,
            }

        return {
            "id": str(document.id),
            "source": document.source,
            "source_document_id": document.source_document_id,
            "filename": document.filename,
            "current_version": document.current_version,
            "status": document.status,
            "created_at": (
                document.created_at.isoformat()
                if document.created_at
                else None
            ),
            "updated_at": (
                document.updated_at.isoformat()
                if document.updated_at
                else None
            ),
        }


@mcp.tool
async def search_company_documents(
    query: str,
    top_k: int = 5,
    source: str | None = None,
    source_document_id: str | None = None,
    include_rankings: bool = False,
) -> dict:
    """
    Search company documents using hybrid retrieval and cross-encoder reranking.

    Optionally returns rankings before and after cross-encoder reranking
    for evaluation purposes.
    """

    if not query or not query.strip():
        return {
            "error": "Search query cannot be empty",
            "results": [],
        }

    if top_k < 1:
        return {
            "error": "top_k must be greater than zero",
            "results": [],
        }

    async with AsyncSessionLocal() as session:

        retrieval_service = RetrievalService(session)

        results = await retrieval_service.search(
            query=query,
            top_k=top_k,
            source=source,
            source_document_id=source_document_id,
            include_rankings=include_rankings,
        )

        if include_rankings:
            return {
                "query": query,
                "total_results": len(results["after_reranking"]),
                "before_reranking": results["before_reranking"],
                "after_reranking": results["after_reranking"],
                "results": results["after_reranking"],
            }

        return {
            "query": query,
            "total_results": len(results),
            "results": results,
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")
