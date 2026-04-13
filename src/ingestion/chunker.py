"""Split raw transcript segments into overlapping LangChain Documents with metadata."""

from __future__ import annotations

import bisect
import uuid
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
)
from src.generation.citations import _timestamp_url

_SPLITTER_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_transcript_parent_child(
    segments: list[dict[str, Any]],
    video_id: str,
    video_title: str,
    parent_size: int = PARENT_CHUNK_SIZE,
    child_size: int = CHILD_CHUNK_SIZE,
    child_overlap: int = CHILD_CHUNK_OVERLAP,
) -> tuple[list[Document], list[Document]]:
    """Produce two aligned lists: parent chunks (~1000 chars) and child chunks (~200 chars).

    Parent chunks carry the full ``"Video: {title} | Content: {text}"`` prefix
    so the LLM sees readable context.  Child chunks contain **only** the raw
    sub-text (no prefix) so their embeddings are dominated by the topical
    content rather than the repetitive title string.

    Each parent is assigned a UUID4 ``doc_id``.  Each child carries the same
    ``video_id``, ``video_title``, ``start_time``, and ``source_url`` metadata
    as its parent, plus a ``parent_id`` field pointing back to the parent's
    ``doc_id``.

    Args:
        segments: Raw transcript segments from :func:`~src.ingestion.youtube.fetch_transcript`.
            Each entry must have ``text`` and ``start_time`` keys.
        video_id: YouTube video ID string.
        video_title: Human-readable title for the video.
        parent_size: Target character count for parent chunks.
        child_size: Target character count for child chunks.
        child_overlap: Overlap in characters between adjacent child chunks.

    Returns:
        A tuple ``(parent_docs, child_docs)`` where:
            - ``parent_docs``: list of ~``parent_size``-char Documents each with
              ``doc_id``, ``chunk_type="parent"``, and the four standard metadata
              fields.
            - ``child_docs``: list of ~``child_size``-char Documents each with
              ``parent_id``, ``chunk_type="child"``, and the four standard metadata
              fields inherited from their parent.
    """
    if not segments:
        return [], []

    # ------------------------------------------------------------------
    # 1. Join all segment texts into one string, recording char offsets
    #    and timestamps for position → timestamp mapping.
    # ------------------------------------------------------------------
    offsets: list[int] = []
    timestamps: list[float] = []
    parts: list[str] = []
    cursor = 0

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        offsets.append(cursor)
        timestamps.append(float(seg["start_time"]))
        parts.append(text)
        cursor += len(text) + 1  # +1 for the space separator

    full_text = " ".join(parts)

    # ------------------------------------------------------------------
    # 2. Split into parent chunks.
    # ------------------------------------------------------------------
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=_SPLITTER_SEPARATORS,
    )
    parent_texts = parent_splitter.split_text(full_text)

    # ------------------------------------------------------------------
    # 3. Build parent Documents with doc_id and timestamp.
    # ------------------------------------------------------------------
    parent_docs: list[Document] = []
    child_docs: list[Document] = []
    search_from = 0

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        separators=_SPLITTER_SEPARATORS,
    )

    for raw_text in parent_texts:
        pos = full_text.find(raw_text, search_from)
        if pos == -1:
            pos = full_text.find(raw_text.strip(), search_from)
        if pos == -1:
            pos = search_from

        idx = bisect.bisect_right(offsets, pos) - 1
        start_time = timestamps[max(idx, 0)]
        doc_id = str(uuid.uuid4())
        source_url = _timestamp_url(video_id, start_time)

        parent_docs.append(
            Document(
                page_content=f"Video: {video_title} | Content: {raw_text}",
                metadata={
                    "video_id": video_id,
                    "video_title": video_title,
                    "start_time": start_time,
                    "source_url": source_url,
                    "doc_id": doc_id,
                    "chunk_type": "parent",
                },
            )
        )

        # ------------------------------------------------------------------
        # 4. Sub-split the raw parent text into child chunks.
        #    No "Video:" prefix — keeps embeddings topic-pure.
        # ------------------------------------------------------------------
        for child_text in child_splitter.split_text(raw_text):
            child_docs.append(
                Document(
                    page_content=child_text,
                    metadata={
                        "video_id": video_id,
                        "video_title": video_title,
                        "start_time": start_time,
                        "source_url": source_url,
                        "parent_id": doc_id,
                        "chunk_type": "child",
                    },
                )
            )

        advance = max(len(raw_text) - CHUNK_OVERLAP, 1)
        search_from = pos + advance

    return parent_docs, child_docs


def chunk_transcript(
    segments: list[dict[str, Any]],
    video_id: str,
    video_title: str,
) -> list[Document]:
    """Convert raw transcript segments into overlapping Documents with metadata.

    Backward-compatible wrapper around :func:`chunk_transcript_parent_child`
    that returns only the parent chunks.  Callers that need both parent and
    child chunks should call :func:`chunk_transcript_parent_child` directly.

    Each Document carries:
        - page_content: the chunk text (prefixed with ``"Video: {title} | Content: "``)
        - metadata.video_id: YouTube video ID
        - metadata.video_title: human-readable video title
        - metadata.start_time: approximate start time (seconds) of the chunk
        - metadata.source_url: clickable YouTube URL with ``&t=`` timestamp
        - metadata.doc_id: UUID4 string (used as key in the parent store)
        - metadata.chunk_type: ``"parent"``

    Args:
        segments: Raw transcript segments from :func:`~src.ingestion.youtube.fetch_transcript`.
            Each entry must have ``text`` and ``start_time`` keys.
        video_id: YouTube video ID string.
        video_title: Human-readable title for the video.

    Returns:
        List of LangChain Documents ready for indexing.
    """
    parent_docs, _ = chunk_transcript_parent_child(
        segments, video_id, video_title,
        parent_size=PARENT_CHUNK_SIZE,
    )
    return parent_docs
