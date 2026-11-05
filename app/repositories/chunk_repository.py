
from uuid import UUID

from sqlalchemy import (
    Integer,
    cast,
    column,
    delete,
    insert,
    update,
    values,
)
from sqlalchemy.ext.asyncio import AsyncSession
from pgvector.sqlalchemy import Vector

from app.models.document_chunk import DocumentChunk


class ChunkRepository:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def replace_chunks(
        self,
        *,
        document_version_id: UUID,
        chunks: list[str],
    ) -> int:

        await self.session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_version_id == document_version_id
            )
        )

        if not chunks:
            return 0

        chunk_records = [
            {
                "document_version_id": document_version_id,
                "chunk_index": index,
                "content": content,
            }
            for index, content in enumerate(chunks)
        ]

        await self.session.execute(
            insert(DocumentChunk).values(chunk_records)
        )

        return len(chunk_records)

    async def update_embeddings(
        self,
        *,
        document_version_id: UUID,
        embeddings: list[list[float]],
    ) -> int:

        if not embeddings:
            return 0

        if any(len(embedding) != 1536 for embedding in embeddings):
            raise ValueError(
                "Every embedding must contain exactly 1536 dimensions"
            )

        embedding_values = values(
            column("chunk_index", Integer),
            column("embedding", Vector(1536)),
            name="embedding_values",
        ).data([
            (index, embedding)
            for index, embedding in enumerate(embeddings)
        ])

        statement = (
            update(DocumentChunk)
            .where(
                DocumentChunk.document_version_id == document_version_id,
                DocumentChunk.chunk_index == embedding_values.c.chunk_index,
            )
            .values(
                embedding=cast(
                    embedding_values.c.embedding,
                    Vector(1536),
                )
            )
        )

        result = await self.session.execute(statement)

        return result.rowcount
