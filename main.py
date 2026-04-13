"""End-to-end CLI for the YouTube Transcript RAG system.

Usage:
    python main.py [YOUTUBE_URL]

If no URL is given as an argument, the script prompts for one interactively.
"""

from __future__ import annotations

import sys
import uuid

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.generation.chains import build_conversational_chain

from src.config import FAISS_INDEX_PATH
from src.indexing.vector_store import clear_index, create_vector_store, index_exists, load_index, save_index, save_parents
from src.ingestion.chunker import chunk_transcript_parent_child
from src.ingestion.youtube import extract_video_id, fetch_transcript, get_video_title


def ingest(url: str) -> tuple[FAISS, str, str, list[Document], list[Document]]:
    """Fetch, chunk, and index a YouTube video using parent-child chunking.

    Child chunks (~200 chars) are embedded in FAISS and indexed in BM25 so
    each embedding is dominated by a single topic.  Parent chunks (~1000 chars)
    are stored in the ``LocalFileStore`` and returned to the LLM for full context.

    Any existing FAISS index and parent store are deleted before indexing so
    only the newly ingested video's chunks are searchable.

    Args:
        url: Any supported YouTube URL format.

    Returns:
        Tuple of ``(vector_store, video_id, video_title, parent_docs, child_docs)``.
    """
    print("\n[1/4] Extracting video ID from URL...")
    video_id = extract_video_id(url)

    print("[2/4] Fetching video title...")
    video_title = get_video_title(video_id)
    print(f"      Title: {video_title}")

    print("[3/4] Fetching transcript...")
    segments = fetch_transcript(url)
    print(f"      {len(segments)} segments retrieved")

    print("[4/4] Chunking and indexing...")
    parent_docs, child_docs = chunk_transcript_parent_child(segments, video_id, video_title)
    print(f"      {len(parent_docs)} parent chunks, {len(child_docs)} child chunks")

    clear_index()
    vector_store = create_vector_store(child_docs)

    save_index(vector_store)
    save_parents(parent_docs)
    print("      Index saved to disk\n")

    return vector_store, video_id, video_title, parent_docs, child_docs


def ask(chain: RunnableWithMessageHistory, question: str, session_id: str) -> str:
    """Run the conversational RAG chain and return the cited answer.

    Args:
        chain: The multi-turn RAG chain built by :func:`build_conversational_chain`.
        question: The user's question string.
        session_id: Unique session identifier so history persists across turns.

    Returns:
        The LLM answer with inline [Source N] citations and an appended
        source list containing clickable YouTube timestamp URLs.
    """
    return chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )


def main() -> None:
    """Entry point: ingest a video then answer a user question."""
    # --- 1. Get URL ---
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter YouTube URL (press Enter to use existing index): ").strip()

    # --- 2. Ingest or load ---
    if url:
        try:
            vector_store, _, video_title, _, _ = ingest(url)
        except Exception as exc:
            print(f"Ingestion failed: {exc}")
            sys.exit(1)
        ready_msg = f"Video ready: \"{video_title}\""
    else:
        if not index_exists():
            print("Error: no URL provided and no existing index found.")
            sys.exit(1)
        print("\nLoading existing index...")
        try:
            vector_store = load_index()
        except Exception as exc:
            print(f"Failed to load index: {exc}")
            sys.exit(1)
        print("Index loaded.\n")
        ready_msg = "Existing index loaded."

    # --- 3. Build chain and start session ---
    session_id = str(uuid.uuid4())
    chain = build_conversational_chain(vector_store)
    print(ready_msg)
    print(f"Session ID: {session_id}")
    print("Type your question (or 'quit' to exit).\n")

    while True:
        question = input("Your question: ").strip()
        if not question or question.lower() in {"quit", "exit", "q"}:
            break

        try:
            answer = ask(chain, question, session_id)
        except Exception as exc:
            print(f"Error generating answer: {exc}")
            continue

        print(f"\n{answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
