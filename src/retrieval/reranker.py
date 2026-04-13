"""Wrap an EnsembleRetriever with ContextualCompressionRetriever for reranking."""

from __future__ import annotations

from typing import Sequence

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever


class SortOnlyReranker(BaseDocumentCompressor):
    """Re-order documents by cross-encoder score without dropping any.

    Unlike :class:`CrossEncoderReranker`, this compressor scores every
    (query, document) pair and returns them sorted by descending score,
    keeping all documents.  This prevents borderline-relevant chunks
    (e.g. conversational transcript text) from being silently discarded
    when the cross-encoder under-scores them due to domain mismatch.

    Attributes:
        model: A :class:`HuggingFaceCrossEncoder` instance used for scoring.
    """

    model: HuggingFaceCrossEncoder

    class Config:
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks = None,
    ) -> list[Document]:
        """Score all documents and return them sorted by relevance, highest first.

        Args:
            documents: Candidate documents from the base retriever.
            query: The user query string used for cross-encoder scoring.
            callbacks: Optional LangChain callbacks (unused).

        Returns:
            All input documents re-ordered by descending cross-encoder score.
        """
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.score(pairs)
        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored]


def build_reranking_retriever(
    base_retriever: BaseRetriever,
) -> ContextualCompressionRetriever:
    """Wrap a retriever with a sort-only cross-encoder reranker.

    Scores each (query, document) pair with ``BAAI/bge-reranker-base`` and
    re-orders all retrieved documents by score — highest first — without
    dropping any.  The LLM receives all chunks in relevance order and can
    ignore low-scoring ones.

    Args:
        base_retriever: Any LangChain retriever whose results will be reranked.

    Returns:
        A :class:`ContextualCompressionRetriever` wrapping the base retriever.
    """
    cross_encoder = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base",
        model_kwargs={"device": "cpu"},
    )
    compressor = SortOnlyReranker(model=cross_encoder)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


