import logging
import os

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)


def get_embeddings() -> Embeddings:
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
    logger.info("Initializing embeddings provider: %s", provider)

    if provider == "ollama":
        return OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))

    if provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

    if provider == "huggingface_api":
        return HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        )

    raise ValueError(f"Unknown embedding provider: {provider}")
