"""In-memory session store for multi-turn chat history."""

from __future__ import annotations

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Module-level store: session_id -> ChatMessageHistory
_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return the chat history for a session, creating it if it does not exist.

    This function is compatible with LangChain's
    :class:`RunnableWithMessageHistory` interface.

    Args:
        session_id: Unique identifier for the conversation session.

    Returns:
        The :class:`ChatMessageHistory` for the given session.
    """
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


def clear_session(session_id: str) -> None:
    """Remove all messages for a given session.

    Args:
        session_id: The session to clear.
    """
    if session_id in _store:
        del _store[session_id]


