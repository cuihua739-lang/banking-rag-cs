"""Multi-strategy document chunking for banking RAG."""

import re
import uuid
from dataclasses import dataclass, field
from config.settings import CHUNK_CONFIGS, SENTENCE_GROUP_SIZES, MAX_CHUNK_CHARS, MIN_CHUNK_CHARS


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str = ""
    doc_title: str = ""
    category: str = ""
    strategy: str = ""
    chunk_index: int = 0
    section_title: str = ""
    entities: list[str] = field(default_factory=list)
    summary: str = ""
    token_count: int = 0


def _estimate_tokens(text: str) -> int:
    return len(text)  # Approximate: 1 char ~= 1 token for Chinese


# ===== Strategy 1: Fixed-size sliding window =====

def fixed_window_chunks(text: str) -> list[Chunk]:
    results = []
    for cfg in CHUNK_CONFIGS:
        size = cfg["size"]
        overlap = cfg["overlap"]
        step = size - overlap
        idx = 0
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                results.append(Chunk(
                    chunk_id=uuid.uuid4().hex,
                    text=chunk_text,
                    strategy=f"fixed_{size}",
                    chunk_index=idx,
                    token_count=_estimate_tokens(chunk_text),
                ))
            idx += 1
            start += step
            if end >= len(text):
                break
    return results


# ===== Strategy 2: Structural (heading-based) =====

_HEADING_PATTERN = re.compile(
    r'(?:^|\n)(#{1,4}\s+[^\n]+|(?:[一二三四五六七八九十]+|[（(]\s*[一二三四五六七八九十]+\s*[）)])[、，,\s]*[^\n]+)'
)


def structural_chunks(text: str) -> list[Chunk]:
    sections = _HEADING_PATTERN.split(text)
    if len(sections) <= 1:
        sections = [text]

    chunks = []
    current_title = ""
    idx = 0

    for i, part in enumerate(sections):
        part = part.strip()
        if not part:
            continue
        if _HEADING_PATTERN.match("\n" + part) or (i > 0 and len(part) < 50 and "\n" not in part):
            current_title = part.lstrip("# ").strip()
            continue

        # Split large sections at paragraph boundaries
        if len(part) > 800:
            paragraphs = re.split(r'\n{2,}', part)
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(para) > 800:
                    # Further split at sentence boundaries
                    sentences = re.split(r'(?<=[。！？；\.\!\?\;])', para)
                    buf = ""
                    for sent in sentences:
                        if len(buf) + len(sent) > 800 and buf:
                            chunks.append(_make_chunk(buf, current_title, idx, "structural"))
                            idx += 1
                            buf = sent
                        else:
                            buf += sent
                    if buf.strip():
                        chunks.append(_make_chunk(buf.strip(), current_title, idx, "structural"))
                        idx += 1
                else:
                    chunks.append(_make_chunk(para, current_title, idx, "structural"))
                    idx += 1
        else:
            if len(part) < MIN_CHUNK_CHARS and chunks:
                chunks[-1].text += "\n" + part
            else:
                chunks.append(_make_chunk(part, current_title, idx, "structural"))
                idx += 1

    return chunks


def _make_chunk(text: str, title: str, idx: int, strategy: str) -> Chunk:
    return Chunk(
        chunk_id=uuid.uuid4().hex,
        text=text,
        strategy=strategy,
        chunk_index=idx,
        section_title=title,
        token_count=_estimate_tokens(text),
    )


# ===== Strategy 3: Sentence-aware grouping =====

def sentence_group_chunks(text: str) -> list[Chunk]:
    sentences = re.split(r'(?<=[。！？；\.\!\?\;])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    results = []

    for group_size in SENTENCE_GROUP_SIZES:
        idx = 0
        i = 0
        while i < len(sentences):
            group = []
            chars = 0
            while i < len(sentences) and len(group) < group_size and chars < MAX_CHUNK_CHARS:
                group.append(sentences[i])
                chars += len(sentences[i])
                i += 1
            chunk_text = "".join(group).strip()
            if chunk_text:
                results.append(Chunk(
                    chunk_id=uuid.uuid4().hex,
                    text=chunk_text,
                    strategy=f"sentence_{group_size}",
                    chunk_index=idx,
                    token_count=_estimate_tokens(chunk_text),
                ))
                idx += 1

    return results


# ===== Combined: run all strategies =====

def chunk_document(text: str) -> list[Chunk]:
    """Run all three chunking strategies and return combined chunks."""
    chunks = []
    chunks.extend(fixed_window_chunks(text))
    chunks.extend(structural_chunks(text))
    chunks.extend(sentence_group_chunks(text))
    return chunks
