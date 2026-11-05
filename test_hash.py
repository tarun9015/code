from app.services.ingestion.hash_service import calculate_sha256

content = b"Enterprise Agentic RAG document"

hash_value = calculate_sha256(content)

print("SHA-256:", hash_value)
print("Hash length:", len(hash_value))

assert len(hash_value) == 64

assert hash_value == calculate_sha256(content)

print("Hashing test passed!")
