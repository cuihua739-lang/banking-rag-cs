from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="银行智能客服 RAG 系统",
    description="多层检索 + 召回融合去重 + Rerank + LLM关联度判断 + 深度检索",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    from services.llm_client import get_client
    from services.embedding_service import get_embedder

    components = {
        "chromadb": True,
        "whoosh": True,
        "knowledge_graph": True,
        "llm_api": get_client() is not None,
    }
    # Check embedding model
    try:
        emb = get_embedder()
        components["embedding_model"] = emb is not None
    except Exception:
        components["embedding_model"] = False

    all_ok = all(v for v in components.values())
    return {
        "status": "ok" if all_ok else "degraded",
        "components": components,
        "version": "1.0.0",
    }


# Routers (some may not exist yet — safe import)
from routers.admin import router as admin_router
app.include_router(admin_router, prefix="/api")

from routers.chat import router as chat_router
app.include_router(chat_router, prefix="/api")

# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
