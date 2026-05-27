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


async def retrieve(query: str) -> tuple[list[str], float]:
    """Return (chunks, top_score). chunks is empty if top_score < min_score."""
    vector = await embed(query)
    results: list[ScoredPoint] = await get_client().search(
        collection_name=settings.qdrant_collection,
        query_vector=vector,
        limit=settings.top_k,
        with_payload=True,
    )
    if not results or results[0].score < settings.min_score:
        return [], results[0].score if results else 0.0
    return [hit.payload.get("text", "") for hit in results if hit.payload], results[0].score
