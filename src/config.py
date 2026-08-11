from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

APP_TITLE = "Local AI Knowledge Assistant"

# file path of existing GGUF model here:
# models/mistral-7b-instruct-v0.1-q4_k_m.gguf
MODEL_PATH = str(
    BASE_DIR / "models" / "mistral-7b-instruct-v0.1-q4_k_m.gguf"
)

# Lightweight and strong general-purpose sentence-transformer.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
MAX_HISTORY_MESSAGES = 8

# Mistral context configuration.
N_CTX = 4096
N_THREADS = 8
N_GPU_LAYERS = -1
MAX_NEW_TOKENS = 512

# Chunking
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
