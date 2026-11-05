
import json
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.retrieval.retrieval_service import RetrievalService
from app.services.rag.rag_service import RAGService
from app.evaluation.generation_evaluator import GenerationEvaluator


class GenerationEvaluationRunner:

    def __init__(
        self,
        session: AsyncSession,
        dataset_path: str = "tests/evaluation/golden_dataset.json",
        threshold: float = 0.7,
        top_k: int = 5,
    ):
        self.session = session
        self.dataset_path = Path(dataset_path)
        self.top_k = top_k
        self.evaluator = GenerationEvaluator(
            threshold=threshold
        )

    def load_dataset(self) -> list[dict]:

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Golden dataset not found: {self.dataset_path}"
            )

        with self.dataset_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            dataset = json.load(file)

        if not isinstance(dataset, list) or not dataset:
            raise ValueError(
                "Golden dataset must contain at least one evaluation case"
            )

        return dataset

    async def run(self) -> dict:

        dataset = self.load_dataset()

        evaluation_results = []

        for item in dataset:

            question = item["question"]
            expected_answer = item["expected_answer"]

            # Step 1: Retrieve relevant chunks.
            retrieval_service = RetrievalService(self.session)

            retrieved_chunks = await retrieval_service.search(
                query=question,
                top_k=self.top_k,
            )

            # Step 2: Extract retrieved context.
            retrieved_contexts = [
                chunk["content"]
                for chunk in retrieved_chunks
                if chunk.get("content")
            ]

            # Step 3: Generate the actual answer.
            rag_service = RAGService()

            try:
                generation_result = await rag_service.generate_answer(
                    question=question,
                    retrieved_contexts=retrieved_contexts,
                )
            finally:
                await rag_service.close()

            actual_answer = generation_result["answer"]

            # Step 4: Evaluate the generated answer.
            metrics = self.evaluator.evaluate(
                question=question,
                actual_answer=actual_answer,
                expected_answer=expected_answer,
                retrieved_contexts=retrieved_contexts,
            )

            evaluation_results.append({
                "id": item["id"],
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "retrieved_chunks": len(retrieved_chunks),
                "metrics": metrics,
            })

        # Step 5: Calculate average metric scores.
        metric_names = list(
            evaluation_results[0]["metrics"].keys()
        )

        average_scores = {}

        for metric_name in metric_names:

            scores = [
                result["metrics"][metric_name]["score"]
                for result in evaluation_results
            ]

            average_scores[metric_name] = round(
                sum(scores) / len(scores),
                4,
            )

        total_metrics = sum(
            len(result["metrics"])
            for result in evaluation_results
        )

        passed_metrics = sum(
            1
            for result in evaluation_results
            for metric in result["metrics"].values()
            if metric["passed"]
        )

        return {
            "total_questions": len(evaluation_results),
            "total_metrics": total_metrics,
            "passed_metrics": passed_metrics,
            "failed_metrics": total_metrics - passed_metrics,
            "overall_passed": passed_metrics == total_metrics,
            "average_scores": average_scores,
            "individual_results": evaluation_results,
        }
