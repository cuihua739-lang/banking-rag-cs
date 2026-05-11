"""sentence-transformers wrapper. Singleton model, batch embedding."""

import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DIM

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Returns list of float vectors."""
    model = get_embedder()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_single(text: str) -> list[float]:
    return embed([text])[0]


def get_embedding_dim() -> int:
    return EMBEDDING_DIM
