
class ReciprocalRankFusion:

    def __init__(self, rank_constant: int = 60):
        if rank_constant < 1:
            raise ValueError("rank_constant must be greater than zero")

        self.rank_constant = rank_constant

    def fuse(
        self,
        result_lists: list[list[dict]],
        top_k: int = 5,
    ) -> list[dict]:

        if top_k < 1:
            raise ValueError("top_k must be greater than zero")

        scores = {}
        chunks = {}

        for results in result_lists:
            for rank, result in enumerate(results, start=1):

                chunk_id = result["chunk_id"]

                rrf_score = 1 / (
                    self.rank_constant + rank
                )

                scores[chunk_id] = (
                    scores.get(chunk_id, 0) + rrf_score
                )

                # Preserve the original chunk metadata.
                chunks[chunk_id] = {
                    **chunks.get(chunk_id, {}),
                    **result,
                }

        ranked_chunks = sorted(
            scores,
            key=lambda chunk_id: scores[chunk_id],
            reverse=True,
        )

        fused_results = []

        for chunk_id in ranked_chunks[:top_k]:
            fused_results.append({
                **chunks[chunk_id],
                "rrf_score": round(scores[chunk_id], 6),
            })

        return fused_results
