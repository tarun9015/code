
def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:

    if k <= 0:
        raise ValueError("k must be greater than zero")

    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    relevant_count = sum(
        1 for item in retrieved if item in relevant_ids
    )

    return relevant_count / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        return 0.0

    retrieved = retrieved_ids[:k]

    relevant_count = sum(
        1 for item in retrieved if item in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def mean_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:

    if not relevant_ids:
        return 0.0

    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1 / rank

    return 0.0


def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> dict:

    return {
        "precision_at_k": precision_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),
        "recall_at_k": recall_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),
        "mrr": mean_reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        ),
    }
