from fastapi.testclient import TestClient

import app
import server

client = TestClient(server.server)


def test_index_status_endpoint():
    app.index_progress["phase"] = "Embedding 2/4 files"
    try:
        resp = client.get("/api/index/status")
        assert resp.status_code == 200
        assert resp.json() == {"phase": "Embedding 2/4 files"}
    finally:
        app.index_progress["phase"] = ""


def test_index_cancel_endpoint():
    resp = client.post("/api/index/cancel")
    assert resp.status_code == 200
    assert resp.json() == {"status": "cancel_requested"}
    assert app.index_progress["cancel"] is True
    app.index_progress["cancel"] = False


def test_info_endpoint():
    resp = client.get("/api/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "provider" in body and "retrieval_k" in body


def test_search_empty_query():
    resp = client.post("/api/search", json={"query": "", "k": 10})
    assert resp.status_code == 200
    assert resp.json() == {"results": []}


def test_file_endpoint(monkeypatch):
    monkeypatch.setattr(server, "read_file", lambda path: {"source": path, "total_lines": 2, "lines": ["a", "b"]})
    resp = client.get("/api/file", params={"path": "auth.py"})
    assert resp.status_code == 200
    assert resp.json()["lines"] == ["a", "b"]


def test_chat_empty_message_returns_done():
    resp = client.post("/api/chat", json={"message": "", "history": [], "mode": "Quick", "summary": ""})
    assert resp.status_code == 200
    assert '{"type": "done"' in resp.text


def test_index_response_includes_files(monkeypatch):
    monkeypatch.setattr(server, "index_repo", lambda source: ("ok", "<html></html>"))
    app.state.files = ["a.py", "b.py"]
    resp = client.post("/api/index", json={"source": "x"})
    assert resp.status_code == 200
    assert resp.json()["files"] == ["a.py", "b.py"]
