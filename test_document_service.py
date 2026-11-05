import asyncio
import uuid

from app.db.session import AsyncSessionLocal
from app.services.ingestion.document_service import DocumentService


async def main():

    source = "test"
    source_document_id = f"policy-{uuid.uuid4()}"
    filename = "leave_policy.txt"

    async with AsyncSessionLocal() as session:

        service = DocumentService(session)

        # Test 1: First upload
        print("\nTEST 1: First upload")

        document, version, created = await service.register_document(
            source=source,
            source_document_id=source_document_id,
            filename=filename,
            file_content=b"Employees receive 20 days of annual leave.",
            file_type="txt",
            s3_key=f"test/{uuid.uuid4()}.txt",
        )

        print("Document ID:", document.id)
        print("Version:", version.version)
        print("Created:", created)

        assert version.version == 1
        assert created is True

        # Test 2: Duplicate upload
        print("\nTEST 2: Duplicate upload")

        document, version, created = await service.register_document(
            source=source,
            source_document_id=source_document_id,
            filename=filename,
            file_content=b"Employees receive 20 days of annual leave.",
            file_type="txt",
            s3_key=f"test/{uuid.uuid4()}.txt",
        )

        print("Version:", version.version)
        print("Created:", created)

        assert version.version == 1
        assert created is False

        # Test 3: Modified document
        print("\nTEST 3: Modified document")

        document, version, created = await service.register_document(
            source=source,
            source_document_id=source_document_id,
            filename=filename,
            file_content=b"Employees receive 25 days of annual leave.",
            file_type="txt",
            s3_key=f"test/{uuid.uuid4()}.txt",
        )

        print("Version:", version.version)
        print("Created:", created)

        assert version.version == 2
        assert created is True

        print("\nAll document registration tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
