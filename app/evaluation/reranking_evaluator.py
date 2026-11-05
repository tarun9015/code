from app.evaluation.retrieval_evaluator import evaluate_retrieval


def evaluate_reranking(
    before_reranking: list[dict],
    after_reranking: list[dict],
    relevant_ids: set[str],
    k: int = 5,
) -> dict:
    """
    Compare retrieval ranking quality before and after
    cross-encoder reranking.
    """

    before_ids = [
        item["chunk_id"]
        for item in before_reranking
    ]

    after_ids = [
        item["chunk_id"]
        for item in after_reranking
    ]

    before_metrics = evaluate_retrieval(
        retrieved_ids=before_ids,
        relevant_ids=relevant_ids,
        k=k,
    )

    after_metrics = evaluate_retrieval(
        retrieved_ids=after_ids,
        relevant_ids=relevant_ids,
        k=k,
    )

    improvement = {
        metric: round(
            after_metrics[metric] - before_metrics[metric],
            4,
        )
        for metric in before_metrics
    }

    return {
        "before_reranking": before_metrics,
        "after_reranking": after_metrics,
        "improvement": improvement,
    }