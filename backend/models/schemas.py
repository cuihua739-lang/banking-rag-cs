from pydantic import BaseModel, Field
from typing import Optional


# ===== Chat =====

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="客户问题")
    session_id: Optional[str] = Field(default=None, max_length=64)


class Citation(BaseModel):
    doc_title: str
    section_title: str = ""
    chunk_id: str
    excerpt: str = ""


class RetrievalInfo(BaseModel):
    round: int
    average_score: float
    chunks_considered: int
    chunks_used: int
    retrieval_time_ms: int


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    retrieval_info: RetrievalInfo
    follow_up_suggestions: list[str] = []


# ===== Admin / Ingestion =====

class IngestDocument(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    category: str = Field(..., description="credit_cards|loans|accounts|fraud|investments|general")
    source: str = ""


class IngestRequest(BaseModel):
    documents: list[IngestDocument]


class IngestResponse(BaseModel):
    ingested_count: int
    total_chunks: int
    kg_entities_added: int
    kg_relations_added: int
    elapsed_ms: int


class IndexStatsResponse(BaseModel):
    vector_count: int
    whoosh_doc_count: int
    kg_nodes: int
    kg_edges: int
    kg_entity_types: dict[str, int] = {}


class ComponentHealth(BaseModel):
    chromadb: bool
    whoosh: bool
    knowledge_graph: bool
    llm_api: bool


class HealthResponse(BaseModel):
    status: str
    components: ComponentHealth
    version: str


class ErrorResponse(BaseModel):
    error: str
    message: str
