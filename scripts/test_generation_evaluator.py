
import asyncio
from pprint import pprint

from app.db.session import AsyncSessionLocal
from app.services.rag.rag_service import RAGService
from app.evaluation.generation_evaluator import GenerationEvaluator


async def main():

    question = "How many annual leaves are employees entitled to?"

    # Step 1: Generate the RAG response.
    async with AsyncSessionLocal() as session:

        rag_service = RAGService(session)

        rag_result = await rag_service.ask(
            question=question,
            top_k=5,
        )

    # Step 2: Extract retrieved contexts.
    retrieved_contexts = rag_result["retrieved_contexts"]

    if not retrieved_contexts:
        raise RuntimeError(
            "No retrieved contexts available for evaluation"
        )

    # Step 3: Initialize DeepEval.
    evaluator = GenerationEvaluator(
        threshold=0.7
    )

    # Step 4: Evaluate generated answer.
    results = evaluator.evaluate(
        question=question,
        actual_answer=rag_result["answer"],
        expected_answer=(
            "Employees are entitled to 25 days of annual leave."
        ),
        retrieved_contexts=retrieved_contexts,
    )

    # Step 5: Display results.
    print("\nRAG GENERATION EVALUATION")
    print("=" * 50)

    print("\nQuestion:")
    print(question)

    print("\nGenerated Answer:")
    print(rag_result["answer"])

    print("\nRetrieved Contexts:")
    pprint(retrieved_contexts)

    print("\nEvaluation Metrics:")
    pprint(results)

    print("\nGeneration evaluation completed!")


if __name__ == "__main__":
    asyncio.run(main())
