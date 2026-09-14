import chromadb
import pytest
from langchain_core.documents import Document

from rag import vectorstore as vs
from rag.vectorstore import (
    IndexCancelledError,
    build_store,
    search_documents,
)


def _doc(source: str, content: str) -> Document:
    return Document(page_content=content, metadata={"source": source, "repo": "r", "repo_url": "u"})


@pytest.fixture
def chroma_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("rag.vectorstore.CHROMA_DIR", str(tmp_path))
    return tmp_path


def _collection_count() -> int:
    client = chromadb.PersistentClient(path=vs.CHROMA_DIR)
    try:
        return client.get_collection(vs.COLLECTION_NAME).count()
    except Exception:
        return 0


class TestIncremental:
    def test_skip_unchanged_documents(self, chroma_dir, fake_embeddings):
        docs = [_doc("app.py", "x = 1\n"), _doc("utils.py", "def login():\n    pass\n")]
        first = build_store(docs, fake_embeddings)
        embed_calls = []

        class Spy:
            def embed_documents(self, texts):
                embed_calls.append(texts)
                return fake_embeddings.embed_documents(texts)

            def embed_query(self, text):
                return fake_embeddings.embed_query(text)

        second = build_store(docs, Spy())
        assert first == second
        assert embed_calls == []  # nothing re-embedded

    def test_changed_file_gets_new_ids(self, chroma_dir, fake_embeddings):
        docs = [_doc("app.py", "x = 1\n")]
        build_store(docs, fake_embeddings)
        manifest_before = _manifest(chroma_dir)

        docs[0] = _doc("app.py", "x = 2\nlonger body\n" * 30)
        build_store(docs, fake_embeddings)
        manifest_after = _manifest(chroma_dir)

        assert manifest_before["app.py"]["ids"] != manifest_after["app.py"]["ids"]
        assert manifest_after["app.py"]["hash"] != manifest_before["app.py"]["hash"]

    def test_removed_source_prunes_stale_chunks(self, chroma_dir, fake_embeddings):
        docs = [_doc("keep.py", "def login():\n    pass\n"), _doc("drop.py", "z = 0\n")]
        build_store(docs, fake_embeddings)
        before = _collection_count()

        build_store([docs[0]], fake_embeddings)
        after = _collection_count()
        remaining = _manifest(chroma_dir)

        assert before > after
        assert "drop.py" not in remaining
        assert after == sum(len(e["ids"]) for e in remaining.values())

    def test_cancel_aborts_queue(self, chroma_dir, fake_embeddings):
        docs = [_doc("a.py", "x\n"), _doc("b.py", "y\n")]
        with pytest.raises(IndexCancelledError):
            build_store(docs, fake_embeddings, should_cancel=lambda: True)

    def test_search_documents_returns_one_hit_per_file(self, chroma_dir, fake_embeddings):
        build_store(
            [
                _doc("auth.py", "def login(user):\n    return token\n"),
                _doc("config.py", "VERSION = '1.0'\n"),
            ],
            fake_embeddings,
        )
        results = search_documents("how does login work?", k=10, embedding_model=fake_embeddings)
        sources = [r["source"] for r in results]
        assert len(sources) == len(set(sources))
        assert results and "auth" in results[0]["source"]

    def test_search_documents_requires_index(self):
        assert search_documents("anything", embedding_model=None) == []


def _manifest(chroma_dir):
    import json
    from pathlib import Path

    return json.loads((Path(chroma_dir) / "manifest.json").read_text(encoding="utf-8"))
