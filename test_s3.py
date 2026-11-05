from app.services.ingestion.s3_service import S3StorageService

s3 = S3StorageService()

content = b"Enterprise Agentic RAG V2 - S3 connectivity test"

key = s3.generate_s3_key(
    source="v2-test",
    source_document_id="connectivity-check",
    filename="test.txt",
)

print("Generated S3 Key:", key)

uploaded_key = s3.upload_file(
    file_content=content,
    s3_key=key,
    content_type="text/plain",
)

print("Upload successful:", uploaded_key)

s3.delete_file(s3_key=key)

print("Temporary test object deleted successfully!")