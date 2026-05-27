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
    all_images = list(dict.fromkeys(
        img
        for hit in results if hit.payload
        for img in hit.payload.get("images", [])
    ))
    # Prefer Big uploads; cap at 6
    big = [i for i in all_images if "/upload/Big/" in i]
    images = (big or all_images)[:6]
    return chunks, images, results[0].score
