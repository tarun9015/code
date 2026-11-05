
from openai import AsyncOpenAI

from app.core.config import settings


class EmbeddingService:

    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured")

        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY
        )

        self.model = settings.OPENAI_EMBEDDING_MODEL
        self.dimensions = settings.EMBEDDING_DIMENSIONS

    async def generate_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("Input contains empty text")

        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
        )

        embeddings = [
            item.embedding
            for item in sorted(
                response.data,
                key=lambda item: item.index,
            )
        ]

        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Embedding count does not match input text count"
            )

        for embedding in embeddings:
            if len(embedding) != self.dimensions:
                raise RuntimeError(
                    f"Expected {self.dimensions} dimensions, "
                    f"received {len(embedding)}"
                )

        return embeddings

    async def close(self) -> None:
        await self.client.close()
