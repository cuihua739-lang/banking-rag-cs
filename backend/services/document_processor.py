"""Document ingestion orchestrator: chunk → embed → index into all 3 stores + extract KG entities."""

import time
from services.chunking_service import chunk_document
from services.embedding_service import embed
from services.vector_store import add_chunks as vs_add
from services.fulltext_search import add_chunks as fts_add
from services.knowledge_graph import add_document as kg_add, save as kg_save
from services.llm_client import get_client
from prompts.entity_extraction import SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE


async def _extract_kg_entities(title: str, category: str, content: str) -> tuple[list[dict], list[dict]]:
    """Use LLM to extract entities and relations from a document."""
    client = get_client()
    if client is None:
        return [], []

    user_msg = USER_MESSAGE_TEMPLATE.format(title=title, category=category, content=content[:3000])
    try:
        response = await client.messages.create(
            model="deepseek-v4-pro",
            max_tokens=2048,
            temperature=0.0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = ""
        for block in response.content:
            if getattr(block, "text", None):
                text += block.text

        # Parse JSON from response
        import re
        import json
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("entities", []), data.get("relations", [])
    except Exception:
        pass
    return [], []


async def ingest_document(doc: dict) -> dict:
    """Ingest a single document: chunk, embed, index into all stores.

    Returns stats dict: chunks_count, kg_entities, kg_relations.
    """
    title = doc.get("title", "")
    content = doc.get("content", "")
    category = doc.get("category", "")
    doc_id = doc.get("id", "")

    # Step 1: Chunking
    chunks = chunk_document(content)

    # Step 2: Embed all chunks
    chunk_texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    embeddings = embed(chunk_texts)

    # Step 3: Build metadata for each chunk
    vs_metadatas = []
    fts_chunks = []
    for c in chunks:
        c.doc_id = doc_id
        c.doc_title = title
        c.category = category
        vs_metadatas.append({
            "doc_id": doc_id,
            "doc_title": title,
            "category": category,
            "strategy": c.strategy,
            "section_title": c.section_title,
        })
        fts_chunks.append({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "doc_id": doc_id,
            "doc_title": title,
            "category": category,
            "section_title": c.section_title,
            "strategy": c.strategy,
        })

    # Step 4: Ingest into vector store and fulltext index
    vs_add(chunk_ids, chunk_texts, embeddings, vs_metadatas)
    fts_add(fts_chunks)

    # Step 5: Extract KG entities for each chunk (batch by doc for efficiency)
    entities, relations = await _extract_kg_entities(title, category, content)
    for c in chunks:
        kg_add(doc_id, c.chunk_id, c.text, entities, relations)
    kg_save()

    return {
        "chunks_count": len(chunks),
        "kg_entities": len(entities),
        "kg_relations": len(relations),
    }


async def ingest_documents(documents: list[dict]) -> dict:
    """Ingest multiple documents. Returns aggregate stats."""
    t0 = time.time()
    total_chunks = 0
    total_entities = 0
    total_relations = 0

    for doc in documents:
        result = await ingest_document(doc)
        total_chunks += result["chunks_count"]
        total_entities += result["kg_entities"]
        total_relations += result["kg_relations"]

    elapsed = int((time.time() - t0) * 1000)
    return {
        "ingested_count": len(documents),
        "total_chunks": total_chunks,
        "kg_entities_added": total_entities,
        "kg_relations_added": total_relations,
        "elapsed_ms": elapsed,
    }
