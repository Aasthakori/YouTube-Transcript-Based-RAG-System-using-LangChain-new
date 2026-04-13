"""Utilities for formatting retrieved chunks with [Source N] citations."""

from __future__ import annotations

from langchain_core.documents import Document

# Phrases that indicate the LLM refused to answer from context.
# When any of these appear in an answer, the source list is suppressed.
_REFUSAL_PHRASES: tuple[str, ...] = (
    "don't have enough information",
    "do not have enough information",
    "does not contain enough information",
    "not enough information",
    "cannot answer",
    "no information available",
    "not discussed in",
    "not mentioned in",
    "context does not",
    "context doesn't",
    "do not provide information",
    "does not provide information",
    "videos do not provide",
    "video does not provide",
    "sources do not provide",
    "no information about",
    "there is no information",
    "there's no information",
)


def _timestamp_url(video_id: str, start_time: float) -> str:
    """Build a clickable YouTube URL that jumps to a specific timestamp.

    Args:
        video_id: The YouTube video ID.
        start_time: Start time in seconds.

    Returns:
        A YouTube URL with the ``t`` query parameter set to the nearest second.
    """
    seconds = int(start_time)
    return f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"


def format_with_sources(documents: list[Document]) -> tuple[str, list[dict]]:
    """Label each document as [Source N] and build a source reference list.

    The formatted context string is meant to be injected directly into the
    RAG prompt so the LLM can cite sources by number.

    Args:
        documents: Retrieved and (optionally) reranked LangChain Documents.

    Returns:
        A tuple of:
            - ``context_str``: newline-separated labelled chunks for the prompt.
            - ``sources``: list of dicts with keys ``n``, ``video_title``,
              ``start_time``, and ``url`` for appending to the final answer.
    """
    context_parts: list[str] = []
    sources: list[dict] = []

    for n, doc in enumerate(documents, start=1):
        meta = doc.metadata
        video_id: str = meta.get("video_id", "")
        video_title: str = meta.get("video_title", "Unknown video")
        start_time: float = float(meta.get("start_time", 0.0))
        url = _timestamp_url(video_id, start_time)

        label = (
            f"[Source {n}] ({video_title} — {int(start_time)}s)\n{doc.page_content}"
        )
        context_parts.append(label)
        sources.append(
            {
                "n": n,
                "video_title": video_title,
                "start_time": start_time,
                "url": url,
            }
        )

    context_str = "\n\n".join(context_parts)
    return context_str, sources


def parse_citations(answer: str, sources: list[dict]) -> str:
    """Append a formatted source list with clickable YouTube timestamp URLs.

    Takes the raw LLM answer (which already contains inline ``[Source N]``
    references) and appends a ``Sources:`` block so readers can jump directly
    to the relevant moment in each video.

    Args:
        answer: The raw answer text produced by the LLM.
        sources: Source metadata list returned by :func:`format_with_sources`.

    Returns:
        The answer with a ``**Sources:**`` block appended.
    """
    if not sources:
        return answer

    # Suppress source list entirely when the LLM refused to answer.
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in _REFUSAL_PHRASES):
        return answer

    # Strip any sources block the LLM may have generated itself.
    for marker in ("\n\n**Sources:**", "\n**Sources:**", "**Sources:**"):
        idx = answer.find(marker)
        if idx != -1:
            answer = answer[:idx]
            break

    lines = ["\n\n**Sources:**"]
    for src in sources:
        minutes, seconds = divmod(int(src["start_time"]), 60)
        timestamp_label = f"{minutes}:{seconds:02d}"
        lines.append(
            f"- [Source {src['n']}] {src['video_title']} @ {timestamp_label} — {src['url']}"
        )

    return answer + "\n".join(lines)
