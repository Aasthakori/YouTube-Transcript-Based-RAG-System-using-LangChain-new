"""LCEL chains for RAG generation."""

from __future__ import annotations

import re

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TEMPERATURE
from src.generation.citations import parse_citations, format_with_sources, _REFUSAL_PHRASES
from src.generation.prompts import condense_prompt, rag_prompt
from src.memory.session_store import get_session_history
from src.retrieval.hybrid import build_hybrid_retriever, build_parent_expansion_retriever
from src.retrieval.reranker import build_reranking_retriever


def _get_llm() -> ChatOllama:
    """Instantiate the Ollama LLM.

    Returns:
        A :class:`ChatOllama` instance configured from environment variables.
    """
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=LLM_TEMPERATURE,
    )


def build_conversational_chain(vector_store: FAISS) -> RunnableWithMessageHistory:
    """Build a multi-turn RAG chain backed by in-memory session history.

    Wraps an inner pipeline with :class:`RunnableWithMessageHistory` so each
    session's messages are automatically loaded and persisted between turns.

    Pipeline per turn::

        {"question": str, "chat_history": [messages]}  ← injected by wrapper
          → condense_chain (only when history is non-empty)
          → hybrid + reranking retriever
          → format_with_sources
          → rag_prompt | ChatOllama | StrOutputParser
          → parse_citations
          → answer str  ← stored as AIMessage by wrapper

    Invoke with::

        chain.invoke(
            {"question": "..."},
            config={"configurable": {"session_id": "<id>"}},
        )

    Args:
        vector_store: A populated FAISS vector store.

    Returns:
        A :class:`RunnableWithMessageHistory` that accepts
        ``{"question": str}`` and a ``session_id`` in config, and returns
        the final answer string with inline ``[Source N]`` citations.
    """
    llm = _get_llm()
    parser = StrOutputParser()
    condense_chain = condense_prompt | llm | parser

    # Build the retriever stack once per session — BM25 indexing is expensive
    # and the corpus doesn't change within a session.
    docs = list(vector_store.docstore._dict.values())
    ensemble = build_hybrid_retriever(vector_store, docs)
    parent_retriever = build_parent_expansion_retriever(ensemble)
    retriever = build_reranking_retriever(parent_retriever)

    def _run(inputs: dict) -> str:
        """Execute one full RAG turn with optional question condensation."""
        chat_history = inputs.get("chat_history", [])
        question = inputs["question"]

        # Rewrite follow-up as standalone question when history exists.
        standalone_q = (
            condense_chain.invoke({"question": question, "chat_history": chat_history})
            if chat_history
            else question
        )

        retrieved = retriever.invoke(standalone_q)
        context_str, sources = format_with_sources(retrieved)

        raw_answer = (rag_prompt | llm | parser).invoke(
            {
                "context": context_str,
                "question": standalone_q,
                "chat_history": chat_history,
            }
        )

        # Filter sources to only those cited inline, or top 3 if none cited.
        answer_body = re.split(r"\*?\*?Sources?:?\*?\*?", raw_answer, maxsplit=1)[0]
        cited_nums = {int(n) for n in re.findall(r"Source\s*(\d+)", answer_body)}
        answer_lower = answer_body.lower()

        if cited_nums:
            filtered_sources = [s for s in sources if s.get("n") in cited_nums]
        elif any(phrase in answer_lower for phrase in _REFUSAL_PHRASES):
            filtered_sources = []
        else:
            filtered_sources = sources[:3]

        return parse_citations(raw_answer, filtered_sources)

    return RunnableWithMessageHistory(
        RunnableLambda(_run),
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
