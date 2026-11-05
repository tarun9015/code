
import asyncio
from pprint import pprint

from app.db.session import AsyncSessionLocal
from app.evaluation.retrieval_runner import RetrievalEvaluationRunner


async def main():

    async with AsyncSessionLocal() as session:

        runner = RetrievalEvaluationRunner(
            session=session,
            dataset_path="tests/evaluation/golden_dataset.json",
            top_k=5,
        )

        results = await runner.run()

        print("\nRETRIEVAL EVALUATION REPORT")
        print("=" * 50)

        pprint(results)

        print("\nAverage Metrics:")

        for metric, score in results["average_metrics"].items():
            print(f"{metric}: {score:.4f}")

        print("\nRetrieval evaluation completed!")


if __name__ == "__main__":
    asyncio.run(main())
