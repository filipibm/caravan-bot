from qdrant_client import AsyncQdrantClient
from qdrant_client.models import ScoredPoint

from .config import settings
from .embeddings import embed

_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=settings.qdrant_url)
    return _client


def _images_for_query(query: str, results: list[ScoredPoint]) -> list[str]:
    """Return images only when the user is asking about a specific model.

    Images are taken exclusively from the top-scoring product page so we never
    mix photos from different models or show photos on general questions.
    """
    q = query.lower()
    asks_for_images = any(w in q for w in ("photo", "picture", "image", "show", "look like", "looks like"))
    mentions_model = any(w in q for w in ("352", "353", "354", "403", "506", "mini 303", "mini 304",
                                           "priene", "karavan-"))

    if not (asks_for_images or mentions_model):
        return []

    # Only use images from the single highest-scoring product page
    for hit in results:
        if not hit.payload:
            continue
        source = hit.payload.get("source", "")
        if "/karavan-" in source:
            imgs = hit.payload.get("images", [])
            big = [i for i in imgs if "/upload/Big/" in i]
            return (big or imgs)[:6]

    return []


async def retrieve(query: str) -> tuple[list[str], list[str], float]:
    """Return (chunks, images, top_score). chunks is empty if top_score < min_score."""
    vector = await embed(query)
    results: list[ScoredPoint] = await get_client().search(
        collection_name=settings.qdrant_collection,
        query_vector=vector,
        limit=settings.top_k,
        with_payload=True,
    )
    if not results or results[0].score < settings.min_score:
        return [], [], results[0].score if results else 0.0

    chunks = [hit.payload.get("text", "") for hit in results if hit.payload]
    images = _images_for_query(query, results)
    return chunks, images, results[0].score
