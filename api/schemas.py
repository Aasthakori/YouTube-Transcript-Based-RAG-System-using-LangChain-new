"""Pydantic request/response schemas for the FastAPI layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint."""

    video_url: str = Field(
        ...,
        description="Full YouTube URL or bare video ID to ingest.",
        examples=["https://www.youtube.com/watch?v=bnExo2z_84o"],
    )


class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint."""

    video_id: str
    video_title: str
    chunk_count: int


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(..., description="The user's question.")
    session_id: str = Field(
        default="default",
        description="Session ID for multi-turn conversation history.",
    )


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""

    answer: str
    session_id: str
    sources: list[dict]


class EvalResponse(BaseModel):
    """Response body for the /evaluate endpoint."""

    metrics: dict


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    ollama: bool
    index_loaded: bool
