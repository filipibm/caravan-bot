import os

from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBED_MODEL", "paraphrase-multilingual-mpnet-base-v2")

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_batch(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts, show_progress_bar=False).tolist()
