
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.document_chunk import DocumentChunk


class BM25RetrievalService:

    def __init__(self):
        self.bm25 = None
        self.chunks = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return text.lower().split()

    def index_documents(self, chunks: list[dict]) -> None:
        if not chunks:
            raise ValueError(
                "Cannot index an empty document collection"
            )

        self.chunks = chunks

        tokenized_corpus = [
            self.tokenize(chunk["content"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be greater than zero")

        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index has not been initialized"
            )

        tokenized_query = self.tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        # Keep documents containing at least one query term.
        matched_indices = [
            index
            for index, chunk in enumerate(self.chunks)
            if set(tokenized_query).intersection(
                self.tokenize(chunk["content"])
            )
        ]

        ranked_indices = sorted(
            matched_indices,
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        return [
            {
                **self.chunks[index],
                "bm25_score": float(scores[index]),
            }
            for index in ranked_indices
        ]

    async def search_database(
        self,
        session: AsyncSession,
        query: str,
        top_k: int = 5,
        source: str | None = None,
        source_document_id: str | None = None,
    ) -> list[dict]:

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be greater than zero")

        # Fetch only active, successfully embedded document chunks.
        statement = (
            select(
                DocumentChunk.id,
                DocumentChunk.document_version_id,
                DocumentChunk.chunk_index,
                DocumentChunk.content,
                Document.source,
                Document.source_document_id,
                Document.filename,
                DocumentVersion.version,
            )
            .join(
                DocumentVersion,
                DocumentVersion.id
                == DocumentChunk.document_version_id,
            )
            .join(
                Document,
                Document.id == DocumentVersion.document_id,
            )
            .where(
                DocumentVersion.version == Document.current_version,
                DocumentVersion.status == "EMBEDDED",
            )
        )

        # Apply optional metadata filters.
        if source is not None:
            statement = statement.where(
                Document.source == source
            )

        if source_document_id is not None:
            statement = statement.where(
                Document.source_document_id == source_document_id
            )

        result = await session.execute(statement)
        rows = result.all()

        if not rows:
            return []

        # Convert database records into BM25 documents.
        chunks = [
            {
                "chunk_id": str(row.id),
                "document_version_id": str(
                    row.document_version_id
                ),
                "chunk_index": row.chunk_index,
                "content": row.content,
                "source": row.source,
                "source_document_id": row.source_document_id,
                "filename": row.filename,
                "version": row.version,
            }
            for row in rows
        ]

        # Build the BM25 index for the current document collection.
        self.index_documents(chunks)

        # Perform keyword-based retrieval.
        return self.search(
            query=query,
            top_k=top_k,
        )
