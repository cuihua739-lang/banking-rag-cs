"""Core retrieval pipeline: Round 1 (search → fuse → dedup) + Round 2 (deep retrieval)."""

import time
from services.embedding_service import embed_single
from services.vector_store import query as dense_query
from services.fulltext_search import search as sparse_search
from services.knowledge_graph import traverse as kg_traverse
from services.fusion import fuse_and_dedup
from config.settings import (
    DENSE_TOP_K, DENSE_THRESHOLD, DENSE_THRESHOLD_DEEP,
    SPARSE_TOP_K, SPARSE_TOP_K_DEEP,
    KG_MAX_HOPS, KG_MAX_HOPS_DEEP,
    FUSION_TOP_CANDIDATES,
    RERANK_TOP_K, RERANK_TOP_K_DEEP,
)


def _enrich_kg_results(kg_results: list[dict], dense_results: list[dict], sparse_results: list[dict]) -> list[dict]:
    """KG results only have chunk_id and score — enrich with text and metadata from other sources."""
    id_to_chunk: dict[str, dict] = {}
    for r in dense_results + sparse_results:
        cid = r.get("chunk_id", "")
        if cid and cid not in id_to_chunk:
            id_to_chunk[cid] = r

    enriched = []
    for kg in kg_results:
        cid = kg.get("chunk_id", "")
        if cid in id_to_chunk:
            existing = id_to_chunk[cid]
            enriched.append({
                **existing,
                "score": kg["score"],
                "source": "kg",
                "matched_entity": kg.get("matched_entity", ""),
            })
        else:
            # KG result that wasn't in dense/sparse — still include with limited info
            enriched.append({
                "chunk_id": cid,
                "text": "",
                "score": kg["score"],
                "source": "kg",
                "doc_id": "",
                "doc_title": "",
                "category": "",
                "section_title": "",
                "strategy": "",
                "matched_entity": kg.get("matched_entity", ""),
            })
    return enriched


async def round1_retrieval(query: str, top_k: int = FUSION_TOP_CANDIDATES) -> dict:
    """Round 1: search across all three sources, fuse, dedup, return candidates.

    Returns dict with:
      - candidates: list of fused/deduped chunks
      - dense_hits, sparse_hits, kg_hits: raw hits per source
      - elapsed_ms
    """
    t0 = time.time()

    # Step 1: Parallel retrieval from 3 sources
    query_embedding = embed_single(query)
    dense_results = dense_query(query_embedding, top_k=DENSE_TOP_K, threshold=DENSE_THRESHOLD)
    sparse_results = sparse_search(query, limit=SPARSE_TOP_K)
    kg_results = kg_traverse(query, max_hops=KG_MAX_HOPS)

    # Step 2: Enrich KG results with text/metadata from dense/sparse results
    kg_enriched = _enrich_kg_results(kg_results, dense_results, sparse_results)

    # Step 3: Fusion + dedup
    candidates = fuse_and_dedup(
        dense_results, sparse_results, kg_enriched, top_k=top_k
    )

    elapsed = int((time.time() - t0) * 1000)
    return {
        "candidates": candidates,
        "dense_hits": len(dense_results),
        "sparse_hits": len(sparse_results),
        "kg_hits": len(kg_results),
        "elapsed_ms": elapsed,
    }


async def round2_retrieval(
    original_query: str,
    expanded_queries: list[str],
    top_k: int = RERANK_TOP_K_DEEP,
) -> dict:
    """Round 2 deep retrieval: query all variants with lower thresholds, expanded KG hops."""
    t0 = time.time()

    all_dense = []
    all_sparse = []
    all_kg = []

    # Search with original + expanded queries with relaxed thresholds
    for q in [original_query] + expanded_queries:
        q_embed = embed_single(q)
        all_dense.extend(dense_query(q_embed, top_k=DENSE_TOP_K, threshold=DENSE_THRESHOLD_DEEP))
        all_sparse.extend(sparse_search(q, limit=SPARSE_TOP_K_DEEP))
        all_kg.extend(kg_traverse(q, max_hops=KG_MAX_HOPS_DEEP))

    # Dedup within each source by chunk_id (keep highest score)
    def dedup_by_id(items: list[dict]) -> list[dict]:
        seen: dict[str, dict] = {}
        for item in items:
            cid = item.get("chunk_id", "")
            if cid and (cid not in seen or item.get("score", 0) > seen[cid].get("score", 0)):
                seen[cid] = item
        return list(seen.values())

    all_dense = dedup_by_id(all_dense)
    all_sparse = dedup_by_id(all_sparse)
    all_kg = dedup_by_id(all_kg)

    # Enrich KG results
    kg_enriched = _enrich_kg_results(all_kg, all_dense, all_sparse)

    # Fusion + dedup with larger candidate set
    candidates = fuse_and_dedup(all_dense, all_sparse, kg_enriched, top_k=top_k)

    elapsed = int((time.time() - t0) * 1000)
    return {
        "candidates": candidates,
        "dense_hits": len(all_dense),
        "sparse_hits": len(all_sparse),
        "kg_hits": len(all_kg),
        "elapsed_ms": elapsed,
        "expanded_queries": expanded_queries,
    }
