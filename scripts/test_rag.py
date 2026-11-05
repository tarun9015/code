
import asyncio
from pprint import pprint

from app.db.session import AsyncSessionLocal
from app.services.rag.rag_service import RAGService


async def main():

    question = "How many annual leaves are employees entitled to?"

    async with AsyncSessionLocal() as session:

        rag_service = RAGService(session)

        result = await rag_service.ask(
            question=question,
            top_k=5,
        )

        print("\nRAG Question:")
        print(result["question"])

        print("\nGenerated Answer:")
        print(result["answer"])

        print("\nSource References:")
        pprint(result["sources"])

        print("\nRetrieved Chunks:")
        print(result["retrieved_chunks"])

        assert result["answer"], "Generated answer is empty"
        assert result["retrieved_chunks"] > 0, "No chunks retrieved"
        assert result["sources"], "No source references returned"

        print("\nEnd-to-end RAG test passed!")


if __name__ == "__main__":
    asyncio.run(main())
