
import asyncio

from app.services.ingestion.embedding_service import EmbeddingService


async def main():
    service = EmbeddingService()

    try:
        texts = [
            "Employees are entitled to 25 days of annual leave.",
            "Leave applications require manager approval.",
        ]

        embeddings = await service.generate_embeddings(texts)

        print("Total embeddings:", len(embeddings))
        print("Embedding dimensions:", len(embeddings[0]))

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert len(embeddings[1]) == 1536

        print("OpenAI embedding test passed!")

    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
