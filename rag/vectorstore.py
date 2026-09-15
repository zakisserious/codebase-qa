import hashlib
import json
import logging
import os
import shutil
from contextlib import suppress
from pathlib import Path

import chromadb
from chromadb.api.client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from rag.code_splitter import split_documents

logger = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "codebase"
MANIFEST_NAME = "manifest.json"


class IndexCancelledError(Exception):
    """Raised when an indexing run is cancelled mid-way."""


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _manifest_path() -> Path:
    return Path(CHROMA_DIR) / MANIFEST_NAME


def _load_manifest() -> dict:
    path = _manifest_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_manifest(manifest: dict) -> None:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    _manifest_path().write_text(json.dumps(manifest), encoding="utf-8")


def _incremental_enabled(incremental: bool | None) -> bool:
    if incremental is not None:
        return incremental
    return os.getenv("INCREMENTAL_INDEX", "true").lower() not in ("false", "0")


def _full_rebuild(
    client,
    documents: list[Document],
    embedding_model: Embeddings,
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> int:
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    with suppress(ValueError):
        client.delete_collection(COLLECTION_NAME)
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    return len(chunks)


def build_store(
    documents: list[Document],
    embedding_model: Embeddings,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    incremental: bool | None = None,
    should_cancel=None,
    progress_cb=None,
) -> int:
    if chunk_size is None:
        chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    if chunk_overlap is None:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if not _incremental_enabled(incremental):
        count = _full_rebuild(client, documents, embedding_model, chunk_size, chunk_overlap)
        logger.info("Full-rebuilt store with %d chunks", count)
        return count

    logger.info("Incremental index: %d documents", len(documents))
    manifest = _load_manifest()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    current_sources = {doc.metadata.get("source", "") for doc in documents}

    stale_ids: list[str] = []
    added = 0
    total_docs = len(documents)
    for n, doc in enumerate(documents, start=1):
        if should_cancel and should_cancel():
            raise IndexCancelledError("Indexing cancelled.")
        if progress_cb:
            progress_cb(n, total_docs)
        source = doc.metadata.get("source", "")
        if not source:
            continue
        content_hash = _hash_text(doc.page_content)
        entry = manifest.get(source)
        if entry and entry.get("hash") == content_hash:
            continue

        chunks = split_documents([doc], chunk_size, chunk_overlap)
        if not chunks:
            continue
        if entry and entry.get("ids"):
            stale_ids.extend(entry["ids"])

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []
        prefix = content_hash[:16]
        for i, chunk in enumerate(chunks):
            ids.append(f"{prefix}-{i}")
            texts.append(chunk.page_content)
            metadatas.append({**chunk.metadata, "doc_hash": content_hash})
        embeddings = embedding_model.embed_documents(texts)
        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        manifest[source] = {"hash": content_hash, "ids": ids}
        added += len(chunks)

    for removed in [s for s in manifest if s not in current_sources]:
        stale_ids.extend(manifest[removed].get("ids") or [])
        del manifest[removed]

    if stale_ids:
        unique_ids = list(dict.fromkeys(stale_ids))
        try:
            collection.delete(ids=unique_ids)
        except Exception as e:
            logger.warning("Failed to delete stale chunks: %s", e)

    total = sum(len(entry.get("ids") or []) for entry in manifest.values())
    if should_cancel and should_cancel():
        raise IndexCancelledError("Indexing cancelled.")
    _save_manifest(manifest)
    logger.info("Indexed %d chunks (new/changed), %d total in store", added, total)
    return total


def get_retriever(k: int = 4, embedding_model: Embeddings | None = None) -> BaseRetriever:
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def search_documents(
    query: str,
    k: int = 10,
    embedding_model: Embeddings | None = None,
) -> list[dict]:
    """Similarity search formatted for the UI. Returns one best chunk per file:
    [{source, start_line, end_line, snippet}]."""
    if not (query or "").strip() or embedding_model is None:
        return []
    retriever = get_retriever(k=max(1, int(k)), embedding_model=embedding_model)
    documents = retriever.invoke(query.strip())
    seen: set[str] = set()
    results: list[dict] = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        if source in seen:
            continue
        seen.add(source)
        results.append(
            {
                "source": source,
                "start_line": doc.metadata.get("start_line", 1),
                "end_line": doc.metadata.get("end_line", 1),
                "snippet": (doc.page_content or "")[:240],
            }
        )
    return results


def clear_store() -> None:
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        logger.info("Cleared ChromaDB store at %s", CHROMA_DIR)
    # The chromadb client caches a System per path with open sqlite handles;
    # without this the next index reuses handles to a deleted database.
    SharedSystemClient.clear_system_cache()
