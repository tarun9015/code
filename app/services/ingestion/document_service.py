from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.repositories.document_repository import DocumentRepository
from app.services.ingestion.hash_service import calculate_sha256


class DocumentService:

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = DocumentRepository(session)

    async def register_document(
        self,
        *,
        source: str,
        source_document_id: str,
        filename: str,
        file_content: bytes,
        file_type: str,
        s3_key: str,
    ) -> tuple[Document, DocumentVersion, bool]:

        if not file_content:
            raise ValueError("Uploaded document is empty")

        content_hash = calculate_sha256(file_content)

        try:
            return await self._register_document(
                source=source,
                source_document_id=source_document_id,
                filename=filename,
                file_type=file_type,
                s3_key=s3_key,
                content_hash=content_hash,
            )

        except IntegrityError:
            await self.session.rollback()

            # Another request may have created this document first.
            # Retry once against the now-existing document.
            return await self._register_document(
                source=source,
                source_document_id=source_document_id,
                filename=filename,
                file_type=file_type,
                s3_key=s3_key,
                content_hash=content_hash,
            )

    async def _register_document(
        self,
        *,
        source: str,
        source_document_id: str,
        filename: str,
        file_type: str,
        s3_key: str,
        content_hash: str,
    ) -> tuple[Document, DocumentVersion, bool]:

        async with self.session.begin():

            document = await self.repository.get_by_source_id_for_update(
                source=source,
                source_document_id=source_document_id,
            )

            if document is None:
                document = Document(
                    source=source,
                    source_document_id=source_document_id,
                    filename=filename,
                    current_version=1,
                    status="NEW",
                )

                await self.repository.create_document(document)

                version = DocumentVersion(
                    document_id=document.id,
                    version=1,
                    s3_key=s3_key,
                    content_hash=content_hash,
                    file_type=file_type,
                    status="NEW",
                )

                await self.repository.create_version(version)
                await self.session.flush()

                return document, version, True

            existing_version = await self.repository.get_version_by_hash(
                document_id=document.id,
                content_hash=content_hash,
            )

            if existing_version:
                return document, existing_version, False

            next_version = document.current_version + 1

            version = DocumentVersion(
                document_id=document.id,
                version=next_version,
                s3_key=s3_key,
                content_hash=content_hash,
                file_type=file_type,
                status="NEW",
            )

            await self.repository.create_version(version)

            document.current_version = next_version
            document.filename = filename
            document.status = "NEW"

            await self.session.flush()

            return document, version, True