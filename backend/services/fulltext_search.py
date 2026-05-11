"""Whoosh BM25 full-text search with jieba Chinese tokenizer."""

import os
import shutil
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.analysis import RegexAnalyzer
from whoosh.qparser import MultifieldParser
from config.settings import WHOOSH_DIR, SPARSE_TOP_K, SPARSE_TOP_K_DEEP

try:
    import jieba
    from whoosh.analysis import Token

    class _JiebaAnalyzer:
        """Analyzer that uses jieba for Chinese word segmentation."""

        def __call__(self, text: str, **kwargs) -> list[Token]:
            tokens = jieba.lcut(text)
            pos = kwargs.get('positions', False)
            result = []
            for i, t in enumerate(tokens):
                t = t.strip()
                if t:
                    token = Token()
                    token.text = t
                    if pos:
                        token.pos = i
                    result.append(token)
            return result

    _analyzer = _JiebaAnalyzer()
except ImportError:
    _analyzer = RegexAnalyzer()

_SCHEMA = Schema(
    chunk_id=ID(stored=True, unique=True),
    text=TEXT(stored=True, analyzer=_analyzer),
    doc_id=ID(stored=True),
    doc_title=TEXT(stored=True),
    category=ID(stored=True),
    section_title=TEXT(stored=True),
    strategy=ID(stored=True),
)


def _get_index() -> index.Index:
    os.makedirs(WHOOSH_DIR, exist_ok=True)
    if not index.exists_in(WHOOSH_DIR):
        idx = index.create_in(WHOOSH_DIR, _SCHEMA)
        idx.close()
    return index.open_dir(WHOOSH_DIR)


def add_chunks(chunks: list[dict]) -> None:
    """Index chunks. Each dict: chunk_id, text, doc_id, doc_title, category, section_title, strategy."""
    if not chunks:
        return
    idx = _get_index()
    writer = idx.writer()
    for ch in chunks:
        writer.update_document(
            chunk_id=ch["chunk_id"],
            text=ch.get("text", ""),
            doc_id=ch.get("doc_id", ""),
            doc_title=ch.get("doc_title", ""),
            category=ch.get("category", ""),
            section_title=ch.get("section_title", ""),
            strategy=ch.get("strategy", ""),
        )
    writer.commit()


def search(query_text: str, limit: int = SPARSE_TOP_K) -> list[dict]:
    """BM25 search. Returns list of {chunk_id, text, score, source, ...}."""
    try:
        idx = _get_index()
    except Exception:
        return []

    with idx.searcher() as searcher:
        parser = MultifieldParser(["text", "doc_title", "section_title"], schema=_SCHEMA)
        q = parser.parse(query_text)
        results = searcher.search(q, limit=limit)
        chunks = []
        for hit in results:
            chunks.append({
                "chunk_id": hit["chunk_id"],
                "text": hit["text"],
                "score": round(hit.score, 4),
                "source": "sparse",
                "doc_id": hit.get("doc_id", ""),
                "doc_title": hit.get("doc_title", ""),
                "category": hit.get("category", ""),
                "section_title": hit.get("section_title", ""),
                "strategy": hit.get("strategy", ""),
            })
        return chunks


def delete_all() -> None:
    if os.path.exists(WHOOSH_DIR):
        shutil.rmtree(WHOOSH_DIR)
        os.makedirs(WHOOSH_DIR)


def count() -> int:
    try:
        idx = _get_index()
        return idx.doc_count()
    except Exception:
        return 0
