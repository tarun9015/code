import boto3
from botocore.exceptions import ClientError, BotoCoreError
from uuid import uuid4

from app.core.config import settings


class S3StorageService:

    def __init__(self):
        self.bucket_name = settings.S3_BUCKET_NAME
        self.region = settings.AWS_REGION

        self.client = boto3.client(
            "s3",
            region_name=self.region,
        )

    def generate_s3_key(
        self,
        *,
        source: str,
        source_document_id: str,
        filename: str,
    ) -> str:
        unique_id = uuid4()

        return (
            f"documents/{source}/"
            f"{source_document_id}/"
            f"{unique_id}/{filename}"
        )

    def upload_file(
        self,
        *,
        file_content: bytes,
        s3_key: str,
        content_type: str,
    ) -> str:

        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME is not configured")

        if not file_content:
            raise ValueError("Cannot upload an empty file")

        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_content,
                ContentType=content_type,
                ServerSideEncryption="AES256",
            )

            return s3_key

        except (ClientError, BotoCoreError) as exc:
            raise RuntimeError(
                f"S3 upload failed: {exc}"
            ) from exc

    def download_file(self, *, s3_key: str, destination_path: str) -> str:
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME is not configured")

        try:
            self.client.download_file(
                self.bucket_name,
                s3_key,
                destination_path,
            )
            return destination_path

        except (ClientError, BotoCoreError) as exc:
            raise RuntimeError(f"S3 download failed: {exc}") from exc

    def delete_file(self, *, s3_key: str) -> None:

        try:
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key,
            )

        except (ClientError, BotoCoreError) as exc:
            raise RuntimeError(
                f"S3 deletion failed: {exc}"
            ) from exc