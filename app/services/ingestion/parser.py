from pathlib import Path
from typing import Any

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorDevice,
    AcceleratorOptions,
)

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat


class DocumentParser:

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".html",
        ".txt",
        ".md",
    }

    def __init__(self):
        pipeline_options = PdfPipelineOptions()

        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU,
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def parse_document(self, file_path: str) -> dict[str, Any]:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        if not path.is_file():
            raise ValueError("Provided path is not a file")

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        try:
            result = self.converter.convert(str(path))
            document = result.document

            extracted_text = document.export_to_markdown()

            if not extracted_text or not extracted_text.strip():
                raise ValueError(
                    "No readable content extracted from document"
                )

            metadata = {
                "filename": path.name,
                "file_type": extension,
                "content_length": len(extracted_text),
                "page_count": len(document.pages),
            }

            return {
                "filename": path.name,
                "content": extracted_text,
                "metadata": metadata,
            }

        except ValueError:
            raise

        except Exception as exc:
            raise RuntimeError(
                f"Document parsing failed: {path.name}"
            ) from exc