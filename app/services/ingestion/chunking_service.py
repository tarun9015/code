from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


class ChunkingService:

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
            length_function=len,
            is_separator_regex=False,
        )

    def split_text(self, text: str) -> list[str]:

        if not text or not text.strip():
            raise ValueError("Cannot split empty document content")

        chunks = self.text_splitter.split_text(text)

        return [
            chunk.strip()
            for chunk in chunks
            if chunk.strip()
        ]