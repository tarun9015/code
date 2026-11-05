
import asyncio

from app.db.session import AsyncSessionLocal
from app.services.retrieval.retrieval_service import RetrievalService


async def main():

    async with AsyncSessionLocal() as session:

        service = RetrievalService(session)

        try:
            results = await service.search(
                query="How many annual leaves are employees entitled to?",
                top_k=5,
            )

            print("\nHybrid Retrieval + Cross-Encoder Results:")
            print(f"Total Results: {len(results)}")

            for rank, result in enumerate(results, start=1):
                print(f"\nRank: {rank}")
                print(f"Chunk ID: {result['chunk_id']}")
                print(f"RRF Score: {result['rrf_score']}")
                print(
                    f"Similarity Score: "
                    f"{result.get('similarity_score')}"
                )
                print(
                    f"BM25 Score: "
                    f"{result.get('bm25_score')}"
                )
                print(
                    f"Reranker Score: "
                    f"{result.get('reranker_score')}"
                )
                print(f"Filename: {result['filename']}")
                print(f"Content: {result['content']}")

            if settings.RERANKER_ENABLED:
                assert all(
                    "reranker_score" in result
                    for result in results
                ), "Reranker scores are missing"

            assert len(results) <= 5

            print("\nRERANKER_TEST_PASSED")

        finally:
            await service.embedding_service.close()


if __name__ == "__main__":
    from app.core.config import settings
    asyncio.run(main())
