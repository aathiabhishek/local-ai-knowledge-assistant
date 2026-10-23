import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

APP_TITLE = "Local AI Knowledge Assistant"

# Local GGUF model path (can be overridden via environment variable in Docker)
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(BASE_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
)

# Optional Cloud API fallback (used ONLY when the local .gguf file is not present on disk)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CLOUD_MODEL_NAME = os.getenv("CLOUD_MODEL_NAME", "openai/gpt-oss-120b")

# Retrieval & Reranking models
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))

# Llama 3.1 context & hardware configuration
N_CTX = int(os.getenv("N_CTX", "8192"))
N_THREADS = int(os.getenv("N_THREADS", "8"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
