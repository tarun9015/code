
from app.evaluation.retrieval_evaluator import evaluate_retrieval


def main():

    # Simulated retrieval results for metric validation.
    retrieved_ids = [
        "chunk_001",
        "chunk_002",
        "chunk_003",
        "chunk_004",
        "chunk_005",
    ]

    relevant_ids = {
        "chunk_001",
        "chunk_003",
    }

    results = evaluate_retrieval(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        k=5,
    )

    print("\nRetrieval Evaluation Results:")

    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

    assert results["precision_at_k"] == 0.4
    assert results["recall_at_k"] == 1.0
    assert results["mrr"] == 1.0

    print("\nRetrieval evaluator test passed!")


if __name__ == "__main__":
    main()
