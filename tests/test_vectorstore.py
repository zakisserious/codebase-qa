from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.vectorstore import build_store, clear_store


class TestBuildStore:
    @patch("rag.vectorstore.chromadb")
    @patch("rag.vectorstore.Chroma")
    @patch("rag.vectorstore.split_documents")
    def test_build_store_returns_count(self, mock_split, mock_chroma, mock_chroma_mod):
        mock_split.return_value = [Document(page_content="x", metadata={"source": "a.py"})]
        mock_client = MagicMock()
        mock_chroma_mod.PersistentClient.return_value = mock_client

        count = build_store([Document(page_content="x", metadata={"source": "a.py"})], MagicMock())
        assert count == 1

    @patch("rag.vectorstore.shutil")
    @patch("rag.vectorstore.os.path.exists")
    def test_clear_store(self, mock_exists, mock_shutil):
        mock_exists.return_value = True
        clear_store()
        mock_shutil.rmtree.assert_called_once()
