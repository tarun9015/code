import hashlib

from app.services.ingestion.hash_service import calculate_sha256


def test_calculate_sha256_returns_correct_hash():
    content = b"Enterprise Agentic RAG"

    expected_hash = hashlib.sha256(content).hexdigest()

    actual_hash = calculate_sha256(content)

    assert actual_hash == expected_hash


def test_calculate_sha256_returns_consistent_hash():
    content = b"Employee Leave Policy"

    first_hash = calculate_sha256(content)
    second_hash = calculate_sha256(content)

    assert first_hash == second_hash


def test_calculate_sha256_handles_empty_content():
    content = b""

    expected_hash = hashlib.sha256(content).hexdigest()

    assert calculate_sha256(content) == expected_hash