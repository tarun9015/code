
import json
from uuid import UUID

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_version import DocumentVersion


class DocumentRepository:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_source_id(
        self,
        source: str,
        source_document_id: str,
    ) -> Document | None:

        query = select(Document).where(
            Document.source == source,
            Document.source_document_id == source_document_id,
        )

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_by_source_id_for_update(
        self,
        source: str,
        source_document_id: str,
    ) -> Document | None:

        query = (
            select(Document)
            .where(
                Document.source == source,
                Document.source_document_id == source_document_id,
            )
            .with_for_update()
        )

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_by_id(
        self,
        document_id: UUID,
    ) -> Document | None:

        query = select(Document).where(
            Document.id == document_id
        )

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def list_documents(
        self,
        *,
        source: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Document]:

        if limit < 1 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")

        if offset < 0:
            raise ValueError("Offset cannot be negative")

        query = select(Document)

        if source is not None:
            query = query.where(
                Document.source == source
            )

        if status is not None:
            query = query.where(
                Document.status == status
            )

        query = (
            query
            .order_by(desc(Document.created_at))
            .offset(offset)
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    async def get_version_by_hash(
        self,
        document_id: UUID,
        content_hash: str,
    ) -> DocumentVersion | None:

        query = select(DocumentVersion).where(
            DocumentVersion.document_id == document_id,
            DocumentVersion.content_hash == content_hash,
        )

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def create_document(
        self,
        document: Document,
    ) -> Document:

        self.session.add(document)
        await self.session.flush()

        return document

    async def create_version(
        self,
        version: DocumentVersion,
    ) -> DocumentVersion:

        self.session.add(version)
        await self.session.flush()

        return version

    async def update_version_status(
        self,
        *,
        version_id: UUID,
        status: str,
    ) -> None:

        version = await self.session.get(
            DocumentVersion,
            version_id,
        )

        if version is None:
            raise ValueError(
                f"Document version not found: {version_id}"
            )

        version.status = status
        await self.session.flush()

    async def update_parser_metadata(
        self,
        *,
        version_id: UUID,
        metadata: dict,
    ) -> None:

        version = await self.session.get(
            DocumentVersion,
            version_id,
        )

        if version is None:
            raise ValueError(
                f"Document version not found: {version_id}"
            )

        version.parser_metadata = json.dumps(metadata)
        await self.session.flush()
