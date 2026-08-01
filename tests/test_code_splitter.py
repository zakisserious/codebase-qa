from langchain_core.documents import Document

from rag.code_splitter import _get_ts_query_nodes, _split_python, split_documents


class TestSplitDocuments:
    def test_python_splitting(self, sample_code_doc):
        chunks = split_documents([sample_code_doc], chunk_size=500)
        assert len(chunks) > 0
        assert all(c.metadata.get("start_line") for c in chunks)
        assert all(c.metadata.get("end_line") for c in chunks)

    def test_fallback_splitting(self):
        doc = Document(
            page_content="Some text content\n" * 100,
            metadata={"source": "readme.md"},
        )
        chunks = split_documents([doc], chunk_size=200)
        assert len(chunks) > 1
        assert all(c.metadata["node_type"] == "text" for c in chunks)

    def test_empty_input(self):
        chunks = split_documents([], chunk_size=1000)
        assert chunks == []

    def test_chunk_size_respected(self):
        doc = Document(
            page_content="def foo():\n    pass\n\ndef bar():\n    pass\n",
            metadata={"source": "funcs.py"},
        )
        chunks = split_documents([doc], chunk_size=30)
        assert len(chunks) >= 1


class TestSplitPython:
    def test_syntax_error_fallback(self):
        doc = Document(
            page_content="def foo(\n  invalid syntax {{{",
            metadata={"source": "bad.py"},
        )
        chunks = _split_python(doc, 1000)
        assert len(chunks) == 1
        assert chunks[0].metadata["node_type"] == "text"

    def test_import_chunking(self):
        doc = Document(
            page_content="import os\nimport sys\n\ndef main():\n    pass\n",
            metadata={"source": "app.py"},
        )
        chunks = _split_python(doc, 1000)
        import_chunks = [c for c in chunks if c.metadata.get("node_type") == "imports"]
        assert len(import_chunks) == 1

    def test_large_function_splitting(self):
        body = "\n".join(f"    line_{i} = {i}" for i in range(100))
        doc = Document(
            page_content=f"def big_function():\n{body}\n",
            metadata={"source": "big.py"},
        )
        chunks = _split_python(doc, 200)
        assert len(chunks) >= 2


class TestTreeSitterQueries:
    def test_javascript_query(self):
        q = _get_ts_query_nodes("javascript")
        assert q is not None
        assert "function" in q

    def test_typescript_query(self):
        q = _get_ts_query_nodes("typescript")
        assert q is not None
        assert "interface" in q

    def test_unknown_lang_returns_none(self):
        assert _get_ts_query_nodes("unknown") is None
