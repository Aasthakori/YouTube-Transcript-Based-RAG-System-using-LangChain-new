"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import api.routes as _routes
from api.routes import router
from src.indexing.vector_store import index_exists, load_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS index on startup if one exists on disk."""
    if index_exists():
        _routes._vector_store = load_index()
    yield
    # Nothing to clean up on shutdown.


app = FastAPI(
    title="YouTube Transcript RAG API",
    description=(
        "Ask questions about YouTube videos. "
        "Answers are grounded in video transcripts with timestamped citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root():
    """Redirect hint for the root path."""
    return {"message": "Visit /docs for the API documentation."}
