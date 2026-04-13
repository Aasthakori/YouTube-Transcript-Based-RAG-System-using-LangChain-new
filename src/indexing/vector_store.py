"""Build, persist, and load the FAISS vector store."""

from __future__ import annotations

from pathlib import Path

from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.storage.file_system import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL, FAISS_INDEX_PATH, PARENT_STORE_PATH


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Instantiate the HuggingFace embedding model from config.

    Returns:
        A :class:`HuggingFaceEmbeddings` instance using the configured model name.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vector_store(chunks: list[Document]) -> FAISS:
    """Embed a list of Documents and create an in-memory FAISS index.

    This function only creates the index — it does **not** persist it to disk.
    Call :func:`save_index` afterwards to write it out.

    Args:
        chunks: LangChain Documents produced by the ingestion pipeline.

    Returns:
        An in-memory :class:`FAISS` vector store ready for querying or saving.

    Raises:
        ValueError: If ``chunks`` is empty.
    """
    if not chunks:
        raise ValueError("Cannot create a vector store from an empty chunk list.")

    embeddings = _get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def save_index(vector_store: FAISS, path: str = str(FAISS_INDEX_PATH)) -> None:
    """Persist a FAISS index to disk.

    Args:
        vector_store: The FAISS vector store to save.
        path: Directory path to write the index files into.
            Defaults to :data:`~src.config.FAISS_INDEX_PATH` from config.
    """
    dest = Path(path)
    dest.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(dest))


def load_index(path: str = str(FAISS_INDEX_PATH)) -> FAISS:
    """Load a previously saved FAISS index from disk.

    Args:
        path: Directory path containing the saved index files.
            Defaults to :data:`~src.config.FAISS_INDEX_PATH` from config.

    Returns:
        The loaded :class:`FAISS` vector store.

    Raises:
        FileNotFoundError: If no index exists at ``path``.
    """
    src_path = Path(path)
    if not src_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found at '{src_path}'. "
            "Run ingestion first to build the index."
        )
    embeddings = _get_embeddings()
    return FAISS.load_local(
        str(src_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def clear_index(
    index_path: str = str(FAISS_INDEX_PATH),
    parent_path: str = str(PARENT_STORE_PATH),
) -> None:
    """Delete the FAISS index and parent store from disk.

    Call this before ingesting a new video when only one video's chunks
    should be searchable at a time.

    Args:
        index_path: Directory containing the saved FAISS index files.
            Defaults to :data:`~src.config.FAISS_INDEX_PATH`.
        parent_path: Directory containing the LocalFileStore parent docs.
            Defaults to :data:`~src.config.PARENT_STORE_PATH`.
    """
    import shutil

    for p in (Path(parent_path), Path(index_path)):
        if p.exists():
            shutil.rmtree(p)


def index_exists(path: str = str(FAISS_INDEX_PATH)) -> bool:
    """Return True if a persisted FAISS index is present on disk.

    Args:
        path: Directory path to check.
            Defaults to :data:`~src.config.FAISS_INDEX_PATH` from config.

    Returns:
        Boolean indicating whether the index directory exists and is non-empty.
    """
    p = Path(path)
    return p.exists() and any(p.iterdir())


def get_parent_store(path: str = str(PARENT_STORE_PATH)) -> BaseStore:
    """Return a LocalFileStore-backed docstore for parent Documents.

    The store directory is created if it does not yet exist.
    ``LocalFileStore`` reads from disk on every ``mget()`` so this function
    is cheap to call repeatedly — no in-memory caching is needed.

    Args:
        path: Directory path for the ``LocalFileStore`` root.
            Defaults to :data:`~src.config.PARENT_STORE_PATH`.

    Returns:
        A ``BaseStore[str, Document]`` backed by the local filesystem.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    byte_store = LocalFileStore(path)
    return create_kv_docstore(byte_store)


def save_parents(
    parent_docs: list[Document],
    path: str = str(PARENT_STORE_PATH),
) -> None:
    """Persist a list of parent Documents to the LocalFileStore.

    Each Document's ``metadata["doc_id"]`` is used as the store key.
    Existing keys are overwritten — safe because doc_ids are UUIDs and
    a new video always produces new UUIDs.

    Args:
        parent_docs: Parent Documents produced by
            :func:`~src.ingestion.chunker.chunk_transcript_parent_child`.
            Each must have ``metadata["doc_id"]`` set.
        path: ``LocalFileStore`` root path.
            Defaults to :data:`~src.config.PARENT_STORE_PATH`.

    Raises:
        ValueError: If any Document is missing ``doc_id`` in its metadata.
    """
    for doc in parent_docs:
        if "doc_id" not in doc.metadata:
            raise ValueError(
                "Parent document is missing 'doc_id' in metadata. "
                "Use chunk_transcript_parent_child() to produce parent docs."
            )
    store = get_parent_store(path)
    store.mset([(doc.metadata["doc_id"], doc) for doc in parent_docs])


def load_parents(
    parent_ids: list[str],
    path: str = str(PARENT_STORE_PATH),
) -> list[Document]:
    """Fetch parent Documents by their doc_ids from the LocalFileStore.

    Args:
        parent_ids: List of ``doc_id`` strings to retrieve.
        path: ``LocalFileStore`` root path.
            Defaults to :data:`~src.config.PARENT_STORE_PATH`.

    Returns:
        List of Documents in the same order as ``parent_ids``.
        IDs that are not found produce ``None`` entries, which are filtered out.
    """
    if not parent_ids:
        return []
    store = get_parent_store(path)
    results = store.mget(parent_ids)
    return [doc for doc in results if doc is not None]


