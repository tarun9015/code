
import json
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.retrieval.retrieval_service import RetrievalService
from app.evaluation.retrieval_evaluator import evaluate_retrieval
from app.evaluation.reranking_evaluator import evaluate_reranking


class RetrievalEvaluationRunner:

    def __init__(
        self,
        session: AsyncSession,
        dataset_path: str = "tests/evaluation/golden_dataset.json",
        top_k: int = 5,
    ):
        self.session = session
        self.dataset_path = Path(dataset_path)
        self.top_k = top_k

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

        if not isinstance(dataset, list):
            raise ValueError("Golden dataset must contain a JSON list")

        if not dataset:
            raise ValueError("Golden dataset contains no evaluation cases")

        return dataset

    async def run(self) -> dict:

        dataset = self.load_dataset()
        evaluation_results = []

        for item in dataset:

            # Each service owns its embedding client lifecycle.
            retrieval_service = RetrievalService(self.session)

            question = item["question"]
            relevant_ids = set(item["relevant_chunk_ids"])

            rankings = await retrieval_service.search(
                query=question,
                top_k=self.top_k,
                include_rankings=True,
            )

            before_results = rankings["before_reranking"]
            after_results = rankings["after_reranking"]

            before_ids = [
                chunk["chunk_id"]
                for chunk in before_results
            ]

            after_ids = [
                chunk["chunk_id"]
                for chunk in after_results
            ]

            retrieval_metrics = evaluate_retrieval(
                retrieved_ids=after_ids,
                relevant_ids=relevant_ids,
                k=self.top_k,
            )

            reranking_metrics = evaluate_reranking(
                before_reranking=before_results,
                after_reranking=after_results,
                relevant_ids=relevant_ids,
                k=self.top_k,
            )

            evaluation_results.append({
                "id": item["id"],
                "question": question,
                "retrieved_chunks_before_reranking": len(before_ids),
                "retrieved_chunks_after_reranking": len(after_ids),
                "retrieval_metrics": retrieval_metrics,
                "reranking_metrics": reranking_metrics,
            })

        metric_names = [
            "precision_at_k",
            "recall_at_k",
            "mrr",
        ]

        average_metrics = {}

        for metric in metric_names:

            average_metrics[metric] = {
                "before_reranking": sum(
                    result["reranking_metrics"]["before_reranking"][metric]
                    for result in evaluation_results
                ) / len(evaluation_results),

                "after_reranking": sum(
                    result["reranking_metrics"]["after_reranking"][metric]
                    for result in evaluation_results
                ) / len(evaluation_results),

                "improvement": sum(
                    result["reranking_metrics"]["improvement"][metric]
                    for result in evaluation_results
                ) / len(evaluation_results),
            }

        return {
            "total_questions": len(evaluation_results),
            "top_k": self.top_k,
            "individual_results": evaluation_results,
            "average_metrics": average_metrics,
        }
