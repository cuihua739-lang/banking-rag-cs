"""ChromaDB vector store for dense retrieval."""

import os
import uuid
import chromadb
from chromadb.config import Settings as ChromaSettings
from config.settings import CHROMA_DIR, CHROMA_COLLECTION, DENSE_TOP_K, DENSE_THRESHOLD, DENSE_THRESHOLD_DEEP

_client = None
_collection = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_chunks(
    chunk_ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict] | None = None,
) -> None:
    if not chunk_ids:
        return
    coll = _get_collection()
    batch_size = 100
    for i in range(0, len(chunk_ids), batch_size):
        end = min(i + batch_size, len(chunk_ids))
        coll.add(
            ids=chunk_ids[i:end],
            documents=texts[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end] if metadatas else None,
        )


def query(
    query_embedding: list[float],
    top_k: int = DENSE_TOP_K,
    threshold: float = DENSE_THRESHOLD,
) -> list[dict]:
    """Search by vector similarity. Returns list of {chunk_id, text, score, metadata}."""
    coll = _get_collection()
    results = coll.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    if results["ids"][0]:
        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            score = 1.0 - distance / 2.0  # Cosine distance range [0,2] → score [0,1]
            if score >= threshold:
                meta = results["metadatas"][0][i] if results["metadatas"][0] else {}
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i],
                    "score": round(score, 4),
                    "source": "dense",
                    "doc_id": meta.get("doc_id", ""),
                    "doc_title": meta.get("doc_title", ""),
                    "category": meta.get("category", ""),
                    "strategy": meta.get("strategy", ""),
                    "section_title": meta.get("section_title", ""),
                })
    return chunks


def delete_all() -> None:
    global _collection
    client = _get_client()
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    _collection = None


def count() -> int:
    try:
        return _get_collection().count()
    except Exception:
        return 0
