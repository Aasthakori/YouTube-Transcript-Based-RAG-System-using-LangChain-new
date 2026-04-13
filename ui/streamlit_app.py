"""Streamlit chat interface — talks to the FastAPI backend."""

from __future__ import annotations

import uuid
import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="YouTube RAG Chat",
    page_icon="▶️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "content": str, "sources": list}
    st.session_state.messages: list[dict] = []

if "current_video" not in st.session_state:
    st.session_state.current_video: dict | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt_ts(seconds: float) -> str:
    """Convert float seconds to MM:SS string."""
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


def _post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Is `uvicorn api.main:app` running?")
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        st.error(f"Error: {detail}")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The server may be overloaded — try again.")
    except Exception as exc:
        st.error(f"Unexpected error: {type(exc).__name__}")
    return None


def _get(path: str, timeout: int = 60) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Is `uvicorn api.main:app` running?")
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        st.error(f"Error: {detail}")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The server may be overloaded — try again.")
    except Exception as exc:
        st.error(f"Unexpected error: {type(exc).__name__}")
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("▶️ YouTube RAG")
    st.markdown("---")

    # --- Ingest ---
    st.subheader("Add a Video")
    video_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )
    if st.button("Ingest Video", use_container_width=True, type="primary"):
        if not video_url.strip():
            st.warning("Paste a YouTube URL first.")
        else:
            with st.spinner("Fetching transcript and building index…"):
                result = _post("/ingest", {"video_url": video_url.strip()})
            if result:
                st.session_state.current_video = result
                st.session_state.messages = []
                st.session_state.session_id = str(uuid.uuid4())
                st.success(
                    f"Indexed **{result['chunk_count']}** chunks from  \n"
                    f"*{result['video_title']}*"
                )

    # --- Current video ---
    if st.session_state.current_video:
        v = st.session_state.current_video
        st.markdown("---")
        st.subheader("Current Video")
        st.markdown(
            f"**{v['video_title']}**  \n"
            f"`{v['video_id']}` — {v['chunk_count']} chunks"
        )

    # --- Clear conversation ---
    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("YouTube Transcript Q&A")

if not st.session_state.current_video:
    st.info("Add a YouTube video in the sidebar to get started.")

# Render conversation history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    ts = _fmt_ts(src.get("start_time", 0))
                    url = src.get("url", "")
                    title = src.get("video_title", "Unknown")
                    n = src.get("n", "?")
                    st.markdown(f"**[Source {n}]** {title} @ {ts} — [▶ {ts}]({url})")

# Chat input.
if prompt := st.chat_input("Ask a question about the videos…"):
    # Show user message immediately.
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call /query and stream response into the assistant bubble.
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = _post(
                "/query",
                {
                    "question": prompt,
                    "session_id": st.session_state.session_id,
                },
            )

        if result:
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        ts = _fmt_ts(src.get("start_time", 0))
                        url = src.get("url", "")
                        title = src.get("video_title", "Unknown")
                        n = src.get("n", "?")
                        st.markdown(f"**[Source {n}]** {title} @ {ts} — [▶ {ts}]({url})")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        else:
            error_msg = "Failed to get a response. Check the API server."
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg, "sources": []}
            )
