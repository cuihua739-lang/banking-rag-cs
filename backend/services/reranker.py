"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3."""

from config.settings import RERANKER_MODEL_NAME, RERANKER_MAX_LENGTH, RERANKER_BATCH_SIZE

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=RERANKER_MAX_LENGTH)
        except Exception:
            _reranker = False
    return _reranker if _reranker is not False else None


def rerank(query: str, chunks: list[dict], top_k: int = 10) -> list[dict]:
    """Re-rank chunks using cross-encoder. Falls back to original order if model unavailable."""
    model = get_reranker()
    if model is None or len(chunks) <= 1:
        return chunks[:top_k]

    pairs = [(query, ch["text"][:RERANKER_MAX_LENGTH]) for ch in chunks]
    try:
        scores = model.predict(pairs, batch_size=RERANKER_BATCH_SIZE, show_progress_bar=False)
        for i, ch in enumerate(chunks):
            ch["rerank_score"] = float(scores[i]) if hasattr(scores, '__iter__') else float(scores)
        chunks.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    except Exception:
        pass

    return chunks[:top_k]
