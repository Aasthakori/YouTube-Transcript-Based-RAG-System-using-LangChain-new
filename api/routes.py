"""FastAPI route definitions."""

from __future__ import annotations

import re
import requests
from fastapi import APIRouter, HTTPException
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from api.schemas import (
    EvalResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from main import ingest
from src.config import OLLAMA_BASE_URL
from src.evaluation.ragas_eval import run_full_evaluation
from src.generation.citations import format_with_sources, parse_citations
from src.generation.chains import _get_llm
from src.generation.prompts import condense_prompt, rag_prompt
from src.memory.session_store import get_session_history
from src.retrieval.hybrid import build_hybrid_retriever, build_parent_expansion_retriever
from src.retrieval.reranker import build_reranking_retriever

router = APIRouter()

# Loaded at startup (api/main.py lifespan) and updated after each /ingest.
_vector_store: FAISS | None = None

# Short continuation phrases that don't need a full condense LLM call.
_CONTINUATION = re.compile(
    r"^(tell\s+me\s+more|explain\s+more|say\s+more|give\s+more|"
    r"elaborate|expand\s+(on\s+that)?|continue|go\s+on|what\s+else|"
    r"more\s+details?|more\s+info(rmation)?|yes|ok|and\??|"
    r"can\s+you\s+elaborate|please\s+elaborate)\s*[.!?]?$",
    re.IGNORECASE,
)

# Regex for detecting refusal phrases in the raw LLM answer.
_REFUSAL_PATTERNS = re.compile(
    r"don.t have enough information|"
    r"does not provide information|"
    r"do not have (enough )?information|"
    r"context does not (contain|include|mention|have)|"
    r"no information (in|from) the",
    re.IGNORECASE,
)


def _check_ollama() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        r = requests.get(OLLAMA_BASE_URL, timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _classify_error(exc: Exception) -> HTTPException:
    """Map a low-level exception to an appropriate HTTPException.

    Checks for Ollama connectivity failures, timeouts, and empty-index errors
    before falling back to a generic 500 with a brief description.

    Args:
        exc: The exception caught by the endpoint handler.

    Returns:
        An :class:`HTTPException` with a user-friendly detail message.
    """
    msg = str(exc)
    if isinstance(exc, (ConnectionError, requests.exceptions.ConnectionError)) or \
            "connection" in msg.lower() or "connect" in msg.lower():
        return HTTPException(
            status_code=503,
            detail="Cannot connect to AI model. Make sure Ollama is running.",
        )
    if isinstance(exc, TimeoutError) or "timeout" in msg.lower() or "timed out" in msg.lower():
        return HTTPException(
            status_code=504,
            detail="The AI is taking too long to respond. Try a shorter or more specific question.",
        )
    if isinstance(exc, ValueError) and "empty" in msg.lower():
        return HTTPException(
            status_code=400,
            detail="No videos have been ingested yet. Add a video first.",
        )
    # Generic fallback — include a brief description but never expose a raw traceback.
    brief = msg.split("\n")[0][:200]
    return HTTPException(
        status_code=500,
        detail=f"An error occurred: {brief}. Check that Ollama is running and try again.",
    )



@router.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    """Return API health status including Ollama reachability and index state."""
    return HealthResponse(
        status="ok",
        ollama=_check_ollama(),
        index_loaded=_vector_store is not None,
    )


@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
def ingest_video(body: IngestRequest) -> IngestResponse:
    """Fetch a YouTube transcript, chunk it, embed it, and build a fresh FAISS index.

    The video title is fetched automatically via yt-dlp — no need to supply it.
    Any existing index is deleted first so only this video's chunks are searchable.
    """
    global _vector_store

    try:
        vector_store, video_id, video_title, parent_docs, child_docs = ingest(body.video_url)
        _vector_store = vector_store
        return IngestResponse(
            video_id=video_id,
            video_title=video_title,
            chunk_count=len(child_docs),
        )
    except HTTPException:
        raise
    except (ConnectionError, requests.exceptions.ConnectionError):
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to AI model. Make sure Ollama is running.",
        )
    except RuntimeError as exc:
        # Surfaces specific messages from fetch_transcript: video unavailable,
        # transcripts disabled, no transcript found, network errors.
        raise HTTPException(
            status_code=422,
            detail=f"Ingestion failed: {exc}",
        ) from exc
    except Exception as exc:
        brief = str(exc).split("\n")[0][:200]
        raise HTTPException(
            status_code=422,
            detail=f"Ingestion failed: {brief}",
        ) from exc


@router.post("/query", response_model=QueryResponse, tags=["query"])
def query(body: QueryRequest) -> QueryResponse:
    """Answer a question using the full RAG pipeline with conversation memory.

    Runs speaker-filtered hybrid retrieval (BM25 + FAISS), cross-encoder
    reranking, and Mistral generation with inline ``[Source N]`` citations.
    Multi-turn history is stored per ``session_id``.

    Returns the answer string and the structured sources list so callers can
    render timestamps and URLs without parsing the answer text.
    """
    try:
        if _vector_store is None:
            raise HTTPException(
                status_code=503,
                detail="No index loaded. POST /ingest at least one video first.",
            )

        llm = _get_llm()
        parser = StrOutputParser()

        history = get_session_history(body.session_id)
        chat_history = history.messages

        standalone_q = body.question
        if chat_history:
            if _CONTINUATION.match(body.question.strip()):
                # Reuse last human question — no extra LLM call.
                last_human = next(
                    (m.content for m in reversed(chat_history) if isinstance(m, HumanMessage)),
                    None,
                )
                standalone_q = last_human if last_human else body.question
            else:
                standalone_q = (condense_prompt | llm | parser).invoke(
                    {"question": body.question, "chat_history": chat_history}
                )

        docs = list(_vector_store.docstore._dict.values())
        ensemble = build_hybrid_retriever(_vector_store, docs)
        parent_retriever = build_parent_expansion_retriever(ensemble)
        retriever = build_reranking_retriever(parent_retriever)
        retrieved = retriever.invoke(standalone_q)

        context_str, sources = format_with_sources(retrieved)

        raw_answer = (rag_prompt | llm | parser).invoke(
            {
                "context": context_str,
                "question": body.question,
                "chat_history": chat_history,
            }
        )

        # Determine which sources to return BEFORE building the Sources block:
        #   1. Answer cites [Source N] → return only cited sources.
        #   2. No [Source N] but not a refusal → return top 3 by relevance.
        #   3. No [Source N] and is a refusal → return empty list.
        # Split off any LLM-generated "Sources:" block before scanning so we
        # don't count source numbers that appear in the appended reference list.
        answer_body = re.split(r"\*?\*?Sources?:?\*?\*?", raw_answer, maxsplit=1)[0]
        cited_nums = {int(n) for n in re.findall(r"Source\s*(\d+)", answer_body)}

        if cited_nums:
            sources = [s for s in sources if s.get("n") in cited_nums]
        elif _REFUSAL_PATTERNS.search(answer_body):
            sources = []
        else:
            sources = sources[:3]

        # Build the final answer with only the filtered sources appended.
        answer = parse_citations(raw_answer, sources)

        # Persist this turn to session history.
        history.add_message(HumanMessage(content=body.question))
        history.add_message(AIMessage(content=answer))

        return QueryResponse(answer=answer, session_id=body.session_id, sources=sources)

    except HTTPException:
        raise
    except RuntimeError as exc:
        # Catches MPS/CUDA OOM and other PyTorch runtime failures.
        raise _classify_error(exc) from exc
    except (ConnectionError, requests.exceptions.ConnectionError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to AI model. Make sure Ollama is running.",
        ) from exc
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="The AI is taking too long to respond. Try a shorter or more specific question.",
        ) from exc
    except ValueError as exc:
        if "empty" in str(exc).lower():
            raise HTTPException(
                status_code=400,
                detail="No videos have been ingested yet. Add a video first.",
            ) from exc
        raise _classify_error(exc) from exc
    except Exception as exc:
        raise _classify_error(exc) from exc


@router.get("/evaluate", response_model=EvalResponse, tags=["evaluation"])
def evaluate() -> EvalResponse:
    """Run the 2-layer deterministic evaluation on all 20 eval questions.

    Layer 1: Hit Rate @4, MRR @4 (retrieval metrics).
    Layer 2: Semantic Similarity, Key Fact Coverage, Refusal Accuracy,
    Citation Coverage, Correct Answer Rate (generation metrics).

    Answers are cached to disk; this endpoint re-uses the cache if available.
    """
    try:
        metrics = run_full_evaluation()
        return EvalResponse(metrics=metrics)
    except HTTPException:
        raise
    except (ConnectionError, requests.exceptions.ConnectionError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to AI model. Make sure Ollama is running.",
        ) from exc
    except Exception as exc:
        brief = str(exc).split("\n")[0][:200]
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {brief}. Check that Ollama is running and try again.",
        ) from exc
