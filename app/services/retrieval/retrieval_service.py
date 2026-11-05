
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from langsmith import traceable

from app.core.config import settings
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.document_chunk import DocumentChunk
from app.services.ingestion.embedding_service import EmbeddingService
from app.services.retrieval.bm25_service import BM25RetrievalService
from app.services.retrieval.reranker import ReciprocalRankFusion
from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker


class RetrievalService:

    def __init__(self, session: AsyncSession):
        self.session = session
        self.embedding_service = EmbeddingService()
        self.bm25_service = BM25RetrievalService()
        self.rrf = ReciprocalRankFusion(rank_constant=60)

        self.cross_encoder = (
            CrossEncoderReranker()
            if settings.RERANKER_ENABLED
            else None
        )

    @traceable(
        name="hybrid_retrieval",
        run_type="chain",
    )
    async def search(
        self,
        query: str,
        top_k: int | None = None,
        source: str | None = None,
        source_document_id: str | None = None,
        include_rankings: bool = False,
    ) -> list[dict] | dict:

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        top_k = (
            settings.RETRIEVAL_TOP_K
            if top_k is None
            else top_k
        )

        if top_k < 1:
            raise ValueError("top_k must be greater than zero")

        candidate_k = max(
            settings.RETRIEVAL_CANDIDATE_K,
            top_k,
        )

        try:
            # Step 1: Generate query embedding.
            embeddings = await self.embedding_service.generate_embeddings(
                [query]
            )
            query_embedding = embeddings[0]

            # Step 2: Build vector search query.
            cosine_distance = DocumentChunk.embedding.cosine_distance(
                query_embedding
            )

            statement = (
                select(
                    DocumentChunk.id,
                    DocumentChunk.document_version_id,
                    DocumentChunk.chunk_index,
                    DocumentChunk.content,
                    Document.source,
                    Document.source_document_id,
                    Document.filename,
                    DocumentVersion.version,
                    cosine_distance.label("distance"),
                )
                .join(
                    DocumentVersion,
                    DocumentVersion.id
                    == DocumentChunk.document_version_id,
                )
                .join(
                    Document,
                    Document.id == DocumentVersion.document_id,
                )
                .where(
                    DocumentVersion.version == Document.current_version,
                    DocumentVersion.status == "EMBEDDED",
                    DocumentChunk.embedding.is_not(None),
                )
            )

            # Step 3: Apply metadata filters.
            if source is not None:
                statement = statement.where(
                    Document.source == source
                )

            if source_document_id is not None:
                statement = statement.where(
                    Document.source_document_id == source_document_id
                )

            statement = (
                statement
                .order_by(cosine_distance)
                .limit(candidate_k)
            )

            result = await self.session.execute(statement)
            rows = result.all()

            vector_results = [
                {
                    "chunk_id": str(row.id),
                    "document_version_id": str(
                        row.document_version_id
                    ),
                    "chunk_index": row.chunk_index,
                    "content": row.content,
                    "source": row.source,
                    "source_document_id": row.source_document_id,
                    "filename": row.filename,
                    "version": row.version,
                    "similarity_score": round(
                        1 - float(row.distance), 4
                    ),
                }
                for row in rows
            ]

            # Step 4: Perform BM25 keyword retrieval.
            bm25_results = await self.bm25_service.search_database(
                session=self.session,
                query=query,
                top_k=candidate_k,
                source=source,
                source_document_id=source_document_id,
            )

            # Step 5: Combine vector and BM25 rankings using RRF.
            hybrid_results = self.rrf.fuse(
                result_lists=[
                    vector_results,
                    bm25_results,
                ],
                top_k=candidate_k,
            )

            # Step 6: Rerank candidates using cross-encoder.
            if self.cross_encoder is not None:
                final_results = self.cross_encoder.rerank(
                    query=query,
                    documents=hybrid_results,
                    top_k=top_k,
                )
            else:
                final_results = hybrid_results[:top_k]

            # Step 7: Optionally return both rankings for evaluation.
            if include_rankings:
                return {
                    "before_reranking": hybrid_results,
                    "after_reranking": final_results,
                }

            return final_results

        finally:
            await self.embedding_service.close()
