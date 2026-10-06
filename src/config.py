from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

APP_TITLE = "Local AI Knowledge Assistant"

# Put the downloaded GGUF file here:
# models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
MODEL_PATH = str(
    BASE_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)

# Lightweight and strong general-purpose sentence-transformer.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
MAX_HISTORY_MESSAGES = 8

# Llama 3.1 context configuration.
N_CTX = 8192
N_THREADS = 8
N_GPU_LAYERS = -1
MAX_NEW_TOKENS = 1024

# Chunking
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

