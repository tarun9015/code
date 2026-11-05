
import logging
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.db.session import get_db
from app.schemas.document import DocumentUploadResponse
from app.services.ingestion.document_service import DocumentService
from app.services.ingestion.hash_service import calculate_sha256
from app.services.ingestion.ingestion_pipeline import IngestionPipeline
from app.services.ingestion.s3_service import S3StorageService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    source: str = Form(...),
    source_document_id: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    file_content = await file.read()

    if not file_content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded document is empty",
        )

    filename = Path(file.filename).name
    file_type = Path(filename).suffix.lstrip(".").lower()

    if not file_type:
        raise HTTPException(
            status_code=400,
            detail="File extension is required",
        )

    content_hash = calculate_sha256(file_content)

    s3_service = S3StorageService()

    s3_key = s3_service.generate_s3_key(
        source=source,
        source_document_id=source_document_id,
        filename=filename,
    )

    # Step 1: Upload original document to S3.
    try:
        await run_in_threadpool(
            s3_service.upload_file,
            file_content=file_content,
            s3_key=s3_key,
            content_type=file.content_type or "application/octet-stream",
        )

    except Exception as exc:
        logger.exception("Document S3 upload failed")

        raise HTTPException(
            status_code=502,
            detail="Document storage failed",
        ) from exc

    # Step 2: Register document and version in PostgreSQL.
    try:
        document, version, created = (
            await DocumentService(db).register_document(
                source=source,
                source_document_id=source_document_id,
                filename=filename,
                file_content=file_content,
                file_type=file_type,
                s3_key=s3_key,
            )
        )

        if not created:
            # Identical content already exists.
            await run_in_threadpool(
                s3_service.delete_file,
                s3_key=s3_key,
            )

    except Exception as exc:
        logger.exception("Document database registration failed")

        try:
            await run_in_threadpool(
                s3_service.delete_file,
                s3_key=s3_key,
            )
        except Exception:
            logger.exception(
                "Failed to clean up S3 object: %s",
                s3_key,
            )

        raise HTTPException(
            status_code=500,
            detail="Document registration failed",
        ) from exc

    # Step 3: Process newly registered document.
    if created:
        try:
            ingestion_pipeline = IngestionPipeline(db)

            ingestion_result = (
                await ingestion_pipeline.process_document(
                    document_version_id=version.id,
                    s3_key=s3_key,
                    filename=filename,
                )
            )

            final_status = ingestion_result["status"]

        except Exception as exc:
            logger.exception(
                "Document ingestion failed: %s",
                version.id,
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "Document uploaded and registered, "
                    "but ingestion failed. Check document status and logs."
                ),
            ) from exc

    else:
        # Do not reprocess identical content.
        final_status = version.status

    return DocumentUploadResponse(
        document_id=document.id,
        document_version_id=version.id,
        filename=document.filename,
        version=version.version,
        status=final_status,
        content_hash=content_hash,
        message=(
            "Document uploaded and processed successfully"
            if created
            else "Document already exists with identical content"
        ),
    )
