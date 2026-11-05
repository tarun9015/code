
import asyncio
from pprint import pprint

from app.db.session import AsyncSessionLocal
from app.services.retrieval.retrieval_service import RetrievalService


async def main():

    query = "How many annual leaves are employees entitled to?"

    async with AsyncSessionLocal() as session:

        retrieval_service = RetrievalService(session)

        results = await retrieval_service.search(
            query=query,
            top_k=5,
        )

        print("\nSearch Query:")
        print(query)

        print("\nRetrieved Results:")
        pprint(results)

        assert len(results) > 0, "No relevant chunks retrieved"

        print("\nSemantic retrieval test passed!")


if __name__ == "__main__":
    asyncio.run(main())
