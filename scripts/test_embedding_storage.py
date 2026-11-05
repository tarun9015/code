
import asyncio
from uuid import UUID

from app.db.session import AsyncSessionLocal
from app.repositories.chunk_repository import ChunkRepository
from app.services.ingestion.embedding_service import EmbeddingService


async def main():
    document_version_id = UUID(
        "5750a004-b56c-4482-b6c9-ca5a42744009"
    )

    embedding_service = EmbeddingService()

    try:
        async with AsyncSessionLocal() as session:
            chunks = [
                "Employees are entitled to 25 days of annual leave."
            ]

            embeddings = await embedding_service.generate_embeddings(
                chunks
            )

            async with session.begin():
                repository = ChunkRepository(session)

                updated_count = await repository.update_embeddings(
                    document_version_id=document_version_id,
                    embeddings=embeddings,
                )

            print("Updated chunks:", updated_count)
            print("Embedding dimensions:", len(embeddings[0]))

            assert updated_count == 1
            print("Embedding storage test passed!")

    finally:
        await embedding_service.close()


if __name__ == "__main__":
    asyncio.run(main())
