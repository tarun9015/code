from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    document_version_id: UUID
    filename: str
    version: int
    status: str
    content_hash: str
    message: str


class DocumentResponse(BaseModel):
    id: UUID
    source: str
    source_document_id: str
    filename: str
    current_version: int
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
