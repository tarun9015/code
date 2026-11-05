
import logging
import tempfile
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository

from app.services.ingestion.parser import DocumentParser
from app.services.ingestion.s3_service import S3StorageService
from app.services.ingestion.chunking_service import ChunkingService
from app.services.ingestion.embedding_service import EmbeddingService


logger = logging.getLogger(__name__)


class IngestionPipeline:

    def __init__(self, session: AsyncSession):
        self.session = session

        self.s3_service = S3StorageService()
        self.parser = DocumentParser()
        self.chunking_service = ChunkingService()
        self.embedding_service = EmbeddingService()

        self.chunk_repository = ChunkRepository(session)
        self.document_repository = DocumentRepository(session)

    async def process_document(
        self,
        *,
        document_version_id: UUID,
        s3_key: str,
        filename: str,
    ) -> dict:

        if not Path(filename).suffix:
            raise ValueError(
                "Document filename must have an extension"
            )

        try:
            # Step 1: Mark document as PROCESSING.
            async with self.session.begin():
                await self.document_repository.update_version_status(
                    version_id=document_version_id,
                    status="PROCESSING",
                )

            # Step 2: Download and parse document.
            with tempfile.TemporaryDirectory(
                prefix="rag_ingestion_"
            ) as temp_dir:

                local_path = Path(temp_dir) / Path(filename).name

                await run_in_threadpool(
                    self.s3_service.download_file,
                    s3_key=s3_key,
                    destination_path=str(local_path),
                )

                parsed_document = await run_in_threadpool(
                    self.parser.parse_document,
                    str(local_path),
                )

            # Step 3: Split extracted text into chunks.
            chunks = self.chunking_service.split_text(
                parsed_document["content"]
            )

            # Step 4: Generate embeddings.
            embeddings = await self.embedding_service.generate_embeddings(
                chunks
            )

            if len(embeddings) != len(chunks):
                raise RuntimeError(
                    "Embedding count does not match chunk count"
                )

            # Step 5: Store chunks, embeddings and metadata.
            async with self.session.begin():

                inserted_count = (
                    await self.chunk_repository.replace_chunks(
                        document_version_id=document_version_id,
                        chunks=chunks,
                    )
                )

                updated_count = (
                    await self.chunk_repository.update_embeddings(
                        document_version_id=document_version_id,
                        embeddings=embeddings,
                    )
                )

                if updated_count != inserted_count:
                    raise RuntimeError(
                        f"Expected {inserted_count} embeddings to be stored, "
                        f"but updated {updated_count} chunks"
                    )

                await self.document_repository.update_parser_metadata(
                    version_id=document_version_id,
                    metadata=parsed_document["metadata"],
                )

                await self.document_repository.update_version_status(
                    version_id=document_version_id,
                    status="EMBEDDED",
                )

            logger.info(
                "Document ingestion completed: %s, chunks: %s, embeddings: %s",
                filename,
                inserted_count,
                updated_count,
            )

            return {
                "filename": filename,
                "document_version_id": str(document_version_id),
                "status": "EMBEDDED",
                "total_chunks": inserted_count,
                "total_embeddings": updated_count,
                "embedding_dimensions": self.embedding_service.dimensions,
                "parser_metadata": parsed_document["metadata"],
            }

        except Exception:

            logger.exception(
                "Document ingestion failed: %s",
                filename,
            )

            # Step 6: Record failure in a separate transaction.
            try:
                async with self.session.begin():
                    await self.document_repository.update_version_status(
                        version_id=document_version_id,
                        status="FAILED",
                    )

            except Exception:
                logger.exception(
                    "Failed to update document status to FAILED: %s",
                    document_version_id,
                )

            raise

        finally:
            try:
                await self.embedding_service.close()
            except Exception:
                logger.exception(
                    "Failed to close embedding service client"
                )
