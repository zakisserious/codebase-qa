import json
import tempfile

from rag.export import export_markdown, export_notebook, export_session


class TestExportMarkdown:
    def test_basic_export(self):
        history = [
            {"role": "user", "content": "What does this do?"},
            {"role": "assistant", "content": "It does things."},
        ]
        md = export_markdown(history, "test-repo")
        assert "# CodeBase QA Session" in md
        assert "test-repo" in md
        assert "What does this do?" in md
        assert "It does things." in md

    def test_empty_history(self):
        md = export_markdown([], "repo")
        assert "# CodeBase QA Session" in md


class TestExportNotebook:
    def test_basic_export(self):
        history = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ]
        nb_json = export_notebook(history, "test-repo")
        nb = json.loads(nb_json)
        assert len(nb["cells"]) >= 3  # header + Q + A

    def test_empty_history(self):
        nb_json = export_notebook([], "repo")
        nb = json.loads(nb_json)
        assert len(nb["cells"]) == 1  # just header


class TestExportSession:
    def test_markdown_export(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", lambda **kwargs: tmp_path / "test.md")
        assert export_session([], "Markdown", "repo") is not None
