#!/usr/bin/env bash
set -e

OLLAMA_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
MODEL="${OLLAMA_MODEL:-mistral}"

# ---------------------------------------------------------------------------
# 1. Wait for Ollama (belt-and-suspenders on top of depends_on: healthy)
# ---------------------------------------------------------------------------
echo "==> Waiting for Ollama at $OLLAMA_URL ..."
until curl -sf "$OLLAMA_URL/api/tags" > /dev/null; do
  echo "    Ollama not ready — retrying in 2s"
  sleep 2
done
echo "==> Ollama is up."

# ---------------------------------------------------------------------------
# 2. Pull model only if not already cached in the ollama_data volume
# ---------------------------------------------------------------------------
if curl -sf "$OLLAMA_URL/api/tags" | grep -q "\"$MODEL\""; then
  echo "==> Model '$MODEL' already present, skipping pull."
else
  echo "==> Pulling model '$MODEL' (~4.1 GB on first run) ..."
  curl -sf -X POST "$OLLAMA_URL/api/pull" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"$MODEL\"}" \
    --no-buffer | tail -1
  echo "==> Model pull complete."
fi

# ---------------------------------------------------------------------------
# 3. Start FastAPI in the background
#    HuggingFace models (all-mpnet-base-v2, bge-reranker-base) download
#    automatically on first /ingest or /query and are cached in the
#    ./data/hf_cache volume for subsequent runs.
# ---------------------------------------------------------------------------
echo "==> Starting FastAPI on port 8000 ..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# ---------------------------------------------------------------------------
# 4. Wait for FastAPI to be ready before starting Streamlit
# ---------------------------------------------------------------------------
echo "==> Waiting for FastAPI to be ready on port 8000 ..."
until curl -sf http://localhost:8000/api/v1/health > /dev/null; do
  echo "    FastAPI not ready — retrying in 1s"
  sleep 1
done
echo "==> FastAPI is ready."

# ---------------------------------------------------------------------------
# 5. Start Streamlit as PID 1 so it receives shutdown signals cleanly
# ---------------------------------------------------------------------------
echo "==> Starting Streamlit on port 8501 ..."
exec streamlit run ui/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true
