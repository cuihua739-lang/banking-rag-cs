"""Admin router: ingestion, stats, and reset endpoints."""

from fastapi import APIRouter
from models.schemas import IngestRequest, IngestResponse, IndexStatsResponse
from services.document_processor import ingest_documents
from services.vector_store import count as vs_count, delete_all as vs_reset
from services.fulltext_search import count as fts_count, delete_all as fts_reset
from services.knowledge_graph import stats as kg_stats, delete_all as kg_reset

router = APIRouter(tags=["admin"])


@router.post("/admin/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Ingest documents: chunk, embed, index into all stores."""
    docs = [d.model_dump() for d in req.documents]
    result = await ingest_documents(docs)
    return IngestResponse(**result)


@router.get("/admin/stats", response_model=IndexStatsResponse)
async def get_stats():
    """Return index statistics from all three stores."""
    kg = kg_stats()
    return IndexStatsResponse(
        vector_count=vs_count(),
        whoosh_doc_count=fts_count(),
        kg_nodes=kg["nodes"],
        kg_edges=kg["edges"],
        kg_entity_types=kg["entity_types"],
    )


@router.delete("/admin/reset")
async def reset():
    """Clear all indexes (vector, fulltext, knowledge graph)."""
    vs_reset()
    fts_reset()
    kg_reset()
    return {"status": "ok", "message": "所有索引已清空"}
