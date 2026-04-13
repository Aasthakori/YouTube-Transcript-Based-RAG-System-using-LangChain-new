"""Build the EnsembleRetriever that combines BM25 sparse + FAISS dense retrieval."""

from __future__ import annotations

from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import BM25_K, FAISS_K, ENSEMBLE_WEIGHTS, PARENT_STORE_PATH


class _ParentExpansionRetriever(BaseRetriever):
    """Wrap a child-chunk retriever so it returns parent Documents instead.

    After the ensemble retriever returns child chunks (each carrying a
    ``parent_id`` in their metadata), this retriever fetches the corresponding
    parent Documents from the ``LocalFileStore`` and returns them deduplicated
    in the order their children ranked.

    This is the workaround for ``ParentDocumentRetriever`` not supporting
    ``EnsembleRetriever`` as its ``vectorstore`` argument.

    Attributes:
        base_retriever: Any retriever that returns child Documents with
            ``parent_id`` in their metadata.
        parent_store_path: Filesystem path to the ``LocalFileStore`` holding
            parent Documents.
    """

    base_retriever: Any
    parent_store_path: str

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Retrieve child chunks then return their parent Documents.

        Args:
            query: The search query string.
            run_manager: LangChain callback manager (passed through).

        Returns:
            Parent Documents in child-ranking order, deduplicated by ``doc_id``.
            Falls back to the child Documents themselves if any child is missing
            a ``parent_id`` (e.g. old-format chunks without parent-child metadata).
        """
        from src.indexing.vector_store import load_parents

        child_docs = self.base_retriever.invoke(query)

        # Collect unique parent_ids preserving rank order.
        # Also record the first (highest-ranked) child content per parent so we
        # can surface it as a "Relevant excerpt" at the top of the parent chunk.
        seen: set[str] = set()
        ordered_ids: list[str] = []
        best_child: dict[str, str] = {}  # pid → child page_content
        fallback_docs: list[Document] = []

        for doc in child_docs:
            pid = doc.metadata.get("parent_id")
            if pid is None:
                # Old-format chunk (no parent_id) — return it directly.
                fallback_docs.append(doc)
            elif pid not in seen:
                seen.add(pid)
                ordered_ids.append(pid)
                best_child[pid] = doc.page_content

        if not ordered_ids:
            return fallback_docs

        parents = load_parents(ordered_ids, self.parent_store_path)
        # Re-sort to preserve child ranking order (load_parents may reorder).
        id_to_parent = {p.metadata["doc_id"]: p for p in parents}

        result: list[Document] = []
        for pid in ordered_ids:
            if pid not in id_to_parent:
                continue
            parent = id_to_parent[pid]
            excerpt = best_child.get(pid, "").strip()
            # Prepend the matched child chunk as a visible excerpt so the LLM
            # sees the most relevant sentence first, even when it is buried deep
            # in the full parent content.
            if excerpt and f'[Relevant excerpt:' not in parent.page_content:
                new_content = (
                    f'[Relevant excerpt: "{excerpt}"]\n\n{parent.page_content}'
                )
                parent = Document(page_content=new_content, metadata=parent.metadata)
            result.append(parent)

        return result + fallback_docs


def build_parent_expansion_retriever(
    base_retriever: BaseRetriever,
    parent_store_path: str = str(PARENT_STORE_PATH),
) -> _ParentExpansionRetriever:
    """Wrap a child-chunk retriever so it returns parent Documents instead.

    Insert this between :func:`build_hybrid_retriever` and
    :func:`~src.retrieval.reranker.build_reranking_retriever` so the reranker
    and LLM receive full ~1000-char parent chunks while retrieval operates on
    focused ~200-char child chunks.

    Args:
        base_retriever: Any retriever returning child Documents with
            ``parent_id`` in their metadata (typically an
            :class:`~langchain_classic.retrievers.ensemble.EnsembleRetriever`
            built from child chunks).
        parent_store_path: Path to the ``LocalFileStore`` holding parent docs.
            Defaults to :data:`~src.config.PARENT_STORE_PATH`.

    Returns:
        A :class:`_ParentExpansionRetriever` whose results are parent Documents.
    """
    return _ParentExpansionRetriever(
        base_retriever=base_retriever,
        parent_store_path=parent_store_path,
    )


def build_hybrid_retriever(
    vector_store: FAISS,
    chunks: list[Document],
    speaker_filter: str | None = None,
) -> EnsembleRetriever:
    """Create an EnsembleRetriever with k=4 per sub-retriever and FAISS-weighted fusion.

    BM25 handles exact keyword matches; FAISS handles semantic similarity.
    Weights are read from :data:`~src.config.ENSEMBLE_WEIGHTS` (default
    [0.3, 0.7]) favouring semantic search while still benefiting from
    keyword recall.

    When ``speaker_filter`` is provided, both BM25 (pre-filtered document list)
    and FAISS (metadata filter on ``video_id``) are restricted to that speaker's
    video, preventing cross-video retrieval errors.

    Args:
        vector_store: A FAISS vector store already populated with embeddings.
        chunks: Full list of chunked Documents used to build the BM25 index.
        speaker_filter: Optional video_id to restrict retrieval to a single
            speaker's video.  Pass ``None`` to retrieve from all videos.

    Returns:
        An :class:`EnsembleRetriever` combining BM25 and FAISS retrieval.
    """
    bm25_docs = (
        [d for d in chunks if d.metadata.get("video_id") == speaker_filter]
        if speaker_filter
        else chunks
    )
    bm25 = BM25Retriever.from_documents(bm25_docs)
    bm25.k = BM25_K

    search_kwargs: dict = {"k": FAISS_K}
    if speaker_filter:
        search_kwargs["filter"] = {"video_id": speaker_filter}
    faiss = vector_store.as_retriever(search_kwargs=search_kwargs)

    return EnsembleRetriever(
        retrievers=[bm25, faiss],
        weights=ENSEMBLE_WEIGHTS,
    )


