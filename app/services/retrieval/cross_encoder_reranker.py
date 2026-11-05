from sentence_transformers import CrossEncoder

from app.core.config import settings


class CrossEncoderReranker:

    def __init__(self):
        self.model = CrossEncoder(
            settings.RERANKER_MODEL
        )

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be greater than zero")

        if not documents:
            return []

        pairs = [
            (query, document["content"])
            for document in documents
        ]

        scores = self.model.predict(pairs)

        reranked_documents = []

        for document, score in zip(documents, scores):
            reranked_documents.append({
                **document,
                "reranker_score": float(score),
            })

        reranked_documents.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        return reranked_documents[:top_k]