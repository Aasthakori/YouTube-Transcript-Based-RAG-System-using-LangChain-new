# YouTube Transcript RAG System

Ask questions about any YouTube video and get cited answers with clickable timestamp links — powered by a fully local, zero-cost RAG pipeline.

## Tech Stack

Mistral 7B (Ollama) · all-mpnet-base-v2 embeddings · FAISS · BM25 · BGE cross-encoder reranker · LangChain LCEL · FastAPI · Streamlit · Docker

## Key Features

- **Parent-child chunking** — 200-char child chunks for precise search, 1000-char parent chunks for LLM context
- **Hybrid retrieval** — BM25 keyword search + FAISS semantic search combined via EnsembleRetriever
- **Cross-encoder reranking** — BGE reranker reads question + chunk together for accurate scoring
- **Conversation memory** — Condense chain resolves pronouns and follow-ups into standalone queries
- **Source citations** — Only cited sources displayed with clickable YouTube timestamp URLs

## Evaluation (RAGAS)

Independent judge: Qwen 3 8B (separate architecture from Mistral 7B generator)

| Metric | Score |
|---|---|
| Faithfulness | **0.89** |
| Response Relevancy | **0.72** |
| Context Precision | **0.86** |
| Refusal Accuracy | **5/6 (83%)** |

Evaluated across 30 questions spanning 3 video transcripts.

## Quick Start

```bash
git clone https://github.com/Aasthakori/YouTube-Transcript-Based-RAG-System-using-LangChain-new.git
cd YouTube-Transcript-Based-RAG-System-using-LangChain-new

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

ollama pull mistral
cp .env.example .env

uvicorn api.main:app --port 8000 &
streamlit run ui/streamlit_app.py
```

Open `http://localhost:8501` → paste a YouTube URL → ask questions.
