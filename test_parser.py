from pprint import pprint
from app.services.ingestion.parser import DocumentParser

parser = DocumentParser()

result = parser.parse_document("test_parser.txt")

print("\nEXTRACTED CONTENT:")
print(result["content"])

print("\nDOCUMENT METADATA:")
pprint(result["metadata"])

assert result["content"]
assert result["metadata"]["content_length"] > 0

print("\nDocument parsing test passed!")