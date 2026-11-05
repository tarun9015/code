
from app.services.retrieval.bm25_service import BM25RetrievalService


documents = [
    {
        "chunk_id": "1",
        "content": "Employees are entitled to 25 days of annual leave.",
    },
    {
        "chunk_id": "2",
        "content": "Employees must submit expense claims within 30 days.",
    },
    {
        "chunk_id": "3",
        "content": "Annual leave applications require manager approval.",
    },
    {
        "chunk_id": "4",
        "content": "Employee salary payments are processed monthly.",
    },
]

bm25_service = BM25RetrievalService()

bm25_service.index_documents(documents)

query = "annual leave approval"

results = bm25_service.search(
    query=query,
    top_k=3,
)

print(f"\nQuery: {query}")
print("\nBM25 Search Results:")

for result in results:
    print(
        f"\nChunk ID: {result['chunk_id']}"
        f"\nBM25 Score: {result['bm25_score']:.4f}"
        f"\nContent: {result['content']}"
    )
