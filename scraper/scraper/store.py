"""Upsert chunks + vectors into Qdrant."""
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_docs")
VECTOR_SIZE = 768    # paraphrase-multilingual-mpnet-base-v2

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    return _client


def ensure_collection() -> None:
    qc = get_client()
    existing = {c.name for c in qc.get_collections().collections}
    if COLLECTION not in existing:
        qc.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def upsert(chunks: list[str], vectors: list[list[float]], source_url: str, images: list[str] | None = None) -> None:
    ensure_collection()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"text": text, "source": source_url, "images": images or []},
        )
        for text, vec in zip(chunks, vectors)
    ]
    get_client().upsert(collection_name=COLLECTION, points=points)
