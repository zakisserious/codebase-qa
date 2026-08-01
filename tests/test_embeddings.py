import os
from unittest.mock import patch

import pytest

from rag.embeddings import get_embeddings


class TestGetEmbeddings:
    def test_ollama_provider(self):
        with (
            patch.dict(os.environ, {"EMBEDDING_PROVIDER": "ollama"}),
            patch("rag.embeddings.OllamaEmbeddings") as mock_cls,
        ):
            get_embeddings()
            mock_cls.assert_called_once()

    def test_huggingface_provider(self):
        with (
            patch.dict(os.environ, {"EMBEDDING_PROVIDER": "huggingface"}),
            patch("rag.embeddings.HuggingFaceEmbeddings") as mock_cls,
        ):
            get_embeddings()
            mock_cls.assert_called_once()

    def test_unknown_provider_raises(self):
        with (
            patch.dict(os.environ, {"EMBEDDING_PROVIDER": "unknown"}),
            pytest.raises(ValueError, match="Unknown embedding provider"),
        ):
            get_embeddings()
