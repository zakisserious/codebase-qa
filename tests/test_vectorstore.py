from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.vectorstore import build_store, clear_store


class TestBuildStore:
    @patch("rag.vectorstore.chromadb")
    def test_build_store_returns_total_count(self, mock_chroma, monkeypatch, tmp_path, fake_embeddings):
        monkeypatch.setattr("rag.vectorstore.CHROMA_DIR", str(tmp_path))
        mock_client = MagicMock()
        mock_chroma.PersistentClient.return_value = mock_client

        count = build_store(
            [Document(page_content="x", metadata={"source": "a.py"})],
            fake_embeddings,
        )
        assert count == 1

    @patch("rag.vectorstore.shutil")
    @patch("rag.vectorstore.os.path.exists")
    def test_clear_store(self, mock_exists, mock_shutil):
        mock_exists.return_value = True
        clear_store()
        mock_shutil.rmtree.assert_called_once()
