"""Central configuration — reads all settings from .env via python-dotenv."""

from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()


def _get(key: str, default: str | None = None) -> str:
    """Return an env variable, raising if it is required but missing."""
    value = os.getenv(key, default)
    if value is None:
        raise EnvironmentError(f"Required environment variable '{key}' is not set.")
    return value


# ---------------------------------------------------------------------------
# Ollama / LLM
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = _get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = _get("OLLAMA_MODEL", "mistral")
LLM_TEMPERATURE: float = float(_get("LLM_TEMPERATURE", "0.2"))

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = _get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(_get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(_get("CHUNK_OVERLAP", "200"))

# Parent-child chunking
# Parent chunks (~1000 chars) are sent to the LLM for full context.
# Child chunks (~200 chars) are embedded in FAISS and indexed in BM25
# so each embedding is dominated by a single topic rather than a mix.
PARENT_CHUNK_SIZE: int = int(_get("PARENT_CHUNK_SIZE", "1000"))
CHILD_CHUNK_SIZE: int = int(_get("CHILD_CHUNK_SIZE", "200"))
CHILD_CHUNK_OVERLAP: int = int(_get("CHILD_CHUNK_OVERLAP", "20"))

# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------
FAISS_INDEX_PATH: Path = Path(_get("FAISS_INDEX_PATH", "faiss_index"))
PARENT_STORE_PATH: Path = FAISS_INDEX_PATH / "parent_store"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
BM25_K: int = int(_get("BM25_K", "16"))   # raised from 6: child chunks need wider net
FAISS_K: int = int(_get("FAISS_K", "16"))  # raised from 6: child chunks need wider net
ENSEMBLE_WEIGHTS: list[float] = [0.3, 0.7]   # [BM25 weight, FAISS weight]
FINAL_K: int = int(_get("FINAL_K", "8"))      # docs kept after reranking (unused by reranker — citation filtering limits user-visible sources)

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST: str = _get("API_HOST", "0.0.0.0")
API_PORT: int = int(_get("API_PORT", "8000"))
