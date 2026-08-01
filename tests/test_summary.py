from unittest.mock import MagicMock

from langchain_core.documents import Document

from rag.summary import _extract_key_files, _parse_response, generate_summary


class TestParseResponse:
    def test_full_response(self):
        text = "SUMMARY: This is a test project.\nTECHNOLOGIES: Python, Flask\nENTRY_POINTS: app.py, main.py"
        result = _parse_response(text)
        assert result["summary"] == "This is a test project."
        assert result["technologies"] == ["Python", "Flask"]
        assert result["entry_points"] == ["app.py", "main.py"]

    def test_partial_response(self):
        text = "SUMMARY: Just a summary."
        result = _parse_response(text)
        assert result["summary"] == "Just a summary."
        assert result["technologies"] == []
        assert result["entry_points"] == []

    def test_empty_response(self):
        result = _parse_response("")
        assert result["summary"] == ""


class TestExtractKeyFiles:
    def test_finds_readme(self):
        docs = [Document(page_content="# README", metadata={"source": "README.md"})]
        result = _extract_key_files(docs)
        assert "README.md" in result

    def test_no_key_files(self):
        docs = [Document(page_content="x", metadata={"source": "random.txt"})]
        result = _extract_key_files(docs)
        assert "No key files" in result


class TestGenerateSummary:
    def test_success(self):
        docs = [Document(page_content="import os", metadata={"source": "app.py"})]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "SUMMARY: A test app.\nTECHNOLOGIES: Python\nENTRY_POINTS: app.py"
        result = generate_summary(docs, mock_llm)
        assert result["description"] == "A test app."
        assert "Python" in result["technologies"]

    def test_llm_failure(self):
        docs = [Document(page_content="x", metadata={"source": "a.py"})]
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM down")
        result = generate_summary(docs, mock_llm)
        assert "Unable to generate" in result["description"]
