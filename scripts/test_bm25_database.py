
import asyncio

from app.db.session import AsyncSessionLocal
from app.services.retrieval.bm25_service import BM25RetrievalService


async def main():

    service = BM25RetrievalService()

    async with AsyncSessionLocal() as session:

        results = await service.search_database(
            session=session,
            query="annual leave entitlement employees",
            top_k=5,
        )

        print("\nPostgreSQL BM25 Search Results:")

        for result in results:
            print(
                f"\nChunk ID: {result['chunk_id']}"
                f"\nBM25 Score: {result['bm25_score']:.4f}"
                f"\nFilename: {result['filename']}"
                f"\nContent: {result['content']}"
            )


if __name__ == "__main__":
    asyncio.run(main())
