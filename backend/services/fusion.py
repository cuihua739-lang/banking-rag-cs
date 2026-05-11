"""Weighted RRF fusion + two-round dedup (MD5 hash + semantic cosine similarity)."""

import hashlib
import numpy as np
from config.settings import (
    RRF_K, FUSION_WEIGHTS, FUSION_TOP_CANDIDATES,
    SEMANTIC_DEDUP_THRESHOLD, EMBEDDING_DIM,
)
from services.embedding_service import embed


def _normalize_text(text: str) -> str:
    """Normalize text for MD5 hashing."""
    return "".join(text.split())


def _md5_hash(text: str) -> str:
    return hashlib.md5(_normalize_text(text).encode("utf-8")).hexdigest()


def weighted_rrf(
    dense_results: list[dict],
    sparse_results: list[dict],
    kg_results: list[dict],
    dense_weight: float = FUSION_WEIGHTS["dense"],
    sparse_weight: float = FUSION_WEIGHTS["sparse"],
    kg_weight: float = FUSION_WEIGHTS["kg"],
    k: int = RRF_K,
) -> list[dict]:
    """Weighted Reciprocal Rank Fusion across three retrieval sources.

    Each result dict must have: chunk_id, text, score, source
    Returns fused list sorted by RRF score descending.
    """
    # Dictionary to accumulate RRF scores and collect metadata
    chunk_map: dict[str, dict] = {}

    sources = [
        (dense_results, dense_weight, "dense"),
        (sparse_results, sparse_weight, "sparse"),
        (kg_results, kg_weight, "kg"),
    ]

    for results, weight, source_name in sources:
        for rank, item in enumerate(results):
            cid = item.get("chunk_id", "")
            if not cid:
                continue
            rrf_score = weight / (k + rank + 1)

            if cid not in chunk_map:
                chunk_map[cid] = {
                    "chunk_id": cid,
                    "text": item.get("text", ""),
                    "rrf_score": 0.0,
                    "sources": [],
                    "raw_scores": {},
                    "doc_id": item.get("doc_id", ""),
                    "doc_title": item.get("doc_title", ""),
                    "category": item.get("category", ""),
                    "section_title": item.get("section_title", ""),
                    "strategy": item.get("strategy", ""),
                    "matched_entity": item.get("matched_entity", ""),
                }

            chunk_map[cid]["rrf_score"] += rrf_score
            chunk_map[cid]["sources"].append(source_name)
            chunk_map[cid]["raw_scores"][source_name] = item.get("score", 0)

    # Sort by RRF score descending
    fused = sorted(chunk_map.values(), key=lambda x: x["rrf_score"], reverse=True)
    return fused


def md5_dedup(chunks: list[dict]) -> list[dict]:
    """Remove exact/similar chunks using MD5 hash of normalized text."""
    seen: set[str] = set()
    result = []
    for ch in chunks:
        h = _md5_hash(ch.get("text", ""))
        if h not in seen:
            seen.add(h)
            result.append(ch)
    return result


def semantic_dedup(chunks: list[dict], threshold: float = SEMANTIC_DEDUP_THRESHOLD) -> list[dict]:
    """Remove semantically similar chunks using cosine similarity.

    When duplicates are found, keep the higher-scored chunk (by RRF).
    """
    if len(chunks) <= 1:
        return chunks

    texts = [ch["text"] for ch in chunks]
    embeddings = np.array(embed(texts))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    embeddings = embeddings / norms

    keep = [True] * len(chunks)
    for i in range(len(chunks)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(chunks)):
            if not keep[j]:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= threshold:
                # Keep the one with higher RRF score
                if chunks[i]["rrf_score"] >= chunks[j]["rrf_score"]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [ch for i, ch in enumerate(chunks) if keep[i]]


def fuse_and_dedup(
    dense_results: list[dict],
    sparse_results: list[dict],
    kg_results: list[dict],
    top_k: int = FUSION_TOP_CANDIDATES,
) -> list[dict]:
    """Full fusion pipeline: RRF → MD5 dedup → semantic dedup → top-k."""
    fused = weighted_rrf(dense_results, sparse_results, kg_results)
    fused = md5_dedup(fused)
    fused = semantic_dedup(fused)
    return fused[:top_k]
