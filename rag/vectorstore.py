import logging
import os
import shutil

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from rag.code_splitter import split_documents

logger = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "codebase"


def build_store(
    documents: list[Document],
    embedding_model: Embeddings,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    logger.info("Splitting %d documents into chunks...", len(documents))
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    logger.info("Created %d chunks", len(chunks))

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.debug("Deleted existing collection '%s'", COLLECTION_NAME)
    except ValueError:
        pass

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    logger.info("Stored %d chunks in ChromaDB at %s", len(chunks), CHROMA_DIR)

    return len(chunks)


def get_retriever(k: int = 4, embedding_model: Embeddings | None = None) -> BaseRetriever:
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def clear_store() -> None:
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        logger.info("Cleared ChromaDB store at %s", CHROMA_DIR)
