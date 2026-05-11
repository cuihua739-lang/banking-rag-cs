import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.environ.get("BANKING_RAG_DATA_DIR", str(BASE_DIR.parent / "banking-rag-data")))

# Anthropic API
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-pro")
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.1

# Embedding
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_DIM = 1024

# Reranker
RERANKER_MODEL_NAME = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_MAX_LENGTH = 512
RERANKER_BATCH_SIZE = 32

# ChromaDB
CHROMA_DIR = str(DATA_DIR / "chroma_db")
CHROMA_COLLECTION = "banking_knowledge"

# Whoosh
WHOOSH_DIR = str(DATA_DIR / "whoosh_index")

# Knowledge Graph
KG_FILE = str(DATA_DIR / "knowledge_graph.json")

# Chunking
CHUNK_CONFIGS = [
    {"name": "fixed_256", "size": 256, "overlap": 51},
    {"name": "fixed_512", "size": 512, "overlap": 102},
    {"name": "fixed_1024", "size": 1024, "overlap": 204},
]
SENTENCE_GROUP_SIZES = [3, 5, 7]
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 100

# Retrieval
DENSE_TOP_K = 20
DENSE_THRESHOLD = 0.5
DENSE_THRESHOLD_DEEP = 0.3
SPARSE_TOP_K = 20
SPARSE_TOP_K_DEEP = 30
KG_MAX_HOPS = 2
KG_MAX_HOPS_DEEP = 3

# Fusion
RRF_K = 60
FUSION_WEIGHTS = {"dense": 1.0, "sparse": 0.8, "kg": 0.6}
FUSION_TOP_CANDIDATES = 30
RERANK_TOP_K = 10
RERANK_TOP_K_DEEP = 15
SEMANTIC_DEDUP_THRESHOLD = 0.95

# Relevance Judgment
RELEVANCE_THRESHOLD = 6.0
RELEVANCE_HIGH_QUALITY = 7
RELEVANCE_MIN_HIGH_COUNT = 2
RELEVANCE_MIN_BEST = 5
RELEVANCE_FALLBACK = 5.0
JUDGE_BATCH_SIZE = 10

# Response
RESPONSE_TOP_CHUNKS = 8
