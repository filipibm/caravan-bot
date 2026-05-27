import asyncio

from sentence_transformers import SentenceTransformer

from .config import settings

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embed_model)
    return _model


async def embed(text: str) -> list[float]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: get_model().encode([text])[0].tolist()
    )
