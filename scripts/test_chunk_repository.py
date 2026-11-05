import asyncio
from uuid import UUID

from app.db.session import AsyncSessionLocal
from app.services.ingestion.ingestion_pipeline import IngestionPipeline


async def main():
    document_version_id = UUID(
        "5750a004-b56c-4482-b6c9-ca5a42744009"
    )

    async with AsyncSessionLocal() as session:
        pipeline = IngestionPipeline(session)

        result = await pipeline.process_document(
            document_version_id=document_version_id,
            s3_key=(
                "documents/test/policy-001/"
                "79ad2482-ba79-4fae-b134-7a859de8ddcc/"
                "test_policy_v2.txt"
            ),
            filename="test_policy_v2.txt",
        )

        print("Ingestion result:", result)

    print("Integrated ingestion test passed!")


if __name__ == "__main__":
    asyncio.run(main())