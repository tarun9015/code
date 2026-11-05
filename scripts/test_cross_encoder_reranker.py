from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker


def main():

    query = "How many annual leaves are employees entitled to?"

    documents = [
        {
            "chunk_id": "1",
            "content": "Employees are entitled to 25 days of annual leave.",
        },
        {
            "chunk_id": "2",
            "content": "Employees receive medical insurance and health benefits.",
        },
        {
            "chunk_id": "3",
            "content": "Annual leave entitlement is 20 working days per year.",
        },
        {
            "chunk_id": "4",
            "content": "The company provides transportation facilities.",
        },
        {
            "chunk_id": "5",
            "content": "Employees must submit expense claims within 30 days.",
        },
    ]

    reranker = CrossEncoderReranker()

    results = reranker.rerank(
        query=query,
        documents=documents,
        top_k=5,
    )

    print("\nCross-Encoder Ranking Results:")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Score: {result['reranker_score']:.4f}")
        print(f"Content: {result['content']}")

    assert results[0]["chunk_id"] in ["1", "3"], (
        "Expected an annual-leave passage to rank first"
    )

    print("\nCROSS_ENCODER_RANKING_TEST_PASSED")


if __name__ == "__main__":
    main()