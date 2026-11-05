import hashlib


def calculate_sha256(file_content: bytes) -> str:
    """
    Calculate SHA-256 hash of document content.

    Args:
        file_content: Raw document bytes.

    Returns:
        Hexadecimal SHA-256 hash.
    """
    sha256 = hashlib.sha256()
    sha256.update(file_content)

    return sha256.hexdigest()