
from app.services.retrieval.reranker import ReciprocalRankFusion


vector_results = [
    {"chunk_id": "1", "content": "Annual leave entitlement is 25 days."},
    {"chunk_id": "2", "content": "Leave requires manager approval."},
    {"chunk_id": "3", "content": "Employees receive monthly salaries."},
]

bm25_results = [
    {"chunk_id": "2", "content": "Leave requires manager approval."},
    {"chunk_id": "1", "content": "Annual leave entitlement is 25 days."},
    {"chunk_id": "4", "content": "Expense claims must be submitted within 30 days."},
]

rrf = ReciprocalRankFusion(rank_constant=60)

results = rrf.fuse(
    result_lists=[vector_results, bm25_results],
    top_k=5,
)

print("\nHybrid RRF Results:")

for result in results:
    print(
        f"Chunk ID: {result['chunk_id']}"
        f" | RRF Score: {result['rrf_score']}"
        f" | Content: {result['content']}"
    )
