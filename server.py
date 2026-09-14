import contextlib
import json
import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import (
    _build_graph,
    ask_question,
    clear_index,
    collect_sources,
    do_export,
    index_progress,
    index_repo,
    read_file,
    restore_index,
    search_documents,
    state,
    stream_deep_analysis,
)

logger = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"


class NoCacheStaticFiles(StaticFiles):
    def __init__(self, directory):
        super().__init__(directory=directory)

    def file_response(
        self,
        full_path,
        stat_result,
        scope,
        status_code: int = 200,
    ):
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["Cache-Control"] = "no-cache"
        return response


def no_cache(response):
    response.headers["Cache-Control"] = "no-cache"
    return response


server = FastAPI(title="CodeBase QA")
server.mount("/static", NoCacheStaticFiles(STATIC), name="static")


class ChatBody(BaseModel):
    message: str = ""
    history: list = []
    mode: str = "Quick"
    summary: str = ""


class IndexBody(BaseModel):
    source: str = ""


class ExportBody(BaseModel):
    format_type: str = "Markdown"
    history: list = []


class SearchBody(BaseModel):
    query: str = ""
    k: int = 10


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@server.get("/")
def index():
    return no_cache(FileResponse(STATIC / "index.html"))


@server.post("/api/chat")
def api_chat(body: ChatBody):
    history = list(body.history)

    def sse():
        if not body.message.strip():
            yield _sse({"type": "done", "history": history, "summary": body.summary})
            return

        last_len = 0
        final_summary = body.summary
        final_history = history

        if body.mode == "Deep Analysis" and state.agent_executor is not None:
            try:
                for kind, payload in stream_deep_analysis(body.message, final_history, final_summary):
                    if kind == "step":
                        yield _sse({"type": "step", "text": payload})
                    elif kind == "answer":
                        if payload:
                            yield _sse({"type": "delta", "text": payload})
                    elif kind == "done":
                        final_history, final_summary = payload
            except Exception as e:
                logger.error("Chat stream error: %s", e)
                yield _sse({"type": "error", "text": f"Error: {e}"})
        else:
            try:
                for updated, new_summary in ask_question(body.message, final_history, body.mode, final_summary):
                    final_history = updated
                    if new_summary:
                        final_summary = new_summary
                    if not updated:
                        continue
                    content = updated[-1].get("content", "")
                    delta = content[last_len:]
                    last_len = len(content)
                    if delta:
                        yield _sse({"type": "delta", "text": delta})
            except Exception as e:
                logger.error("Chat stream error: %s", e)
                yield _sse({"type": "error", "text": f"Error: {e}"})

        try:
            sources = collect_sources(body.message)
            if sources:
                yield _sse({"type": "sources", "items": sources})
        except Exception as e:
            logger.debug("Source collection failed: %s", e)

        yield _sse({"type": "done", "history": final_history, "summary": final_summary})

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@server.post("/api/index")
def api_index(body: IndexBody):
    status, graph_html = index_repo(body.source)
    return {
        "status": status,
        "graph_html": graph_html,
        "files": state.files or [],
    }


@server.post("/api/index/cancel")
def api_index_cancel():
    index_progress["cancel"] = True
    return {"status": "cancel_requested"}


@server.get("/api/index/status")
def api_index_status():
    return {"phase": index_progress.get("phase", "")}


@server.get("/api/info")
def api_info():
    return {
        "provider": os.getenv("LLM_PROVIDER", "ollama"),
        "model": os.getenv("HF_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1")),
        "embedding": os.getenv("EMBEDDING_PROVIDER", "ollama"),
        "retrieval_k": int(os.getenv("RETRIEVAL_K", "4")),
    }


@server.post("/api/search")
def api_search(body: SearchBody):
    if not body.query.strip() or state.embeddings is None:
        return {"results": []}
    results = search_documents(body.query, k=max(1, min(50, body.k)), embedding_model=state.embeddings)
    return {"results": results}


@server.get("/api/file")
def api_file(path: str):
    return read_file(path)


@server.post("/api/clear")
def api_clear():
    status, _, _ = clear_index()
    return {"status": status}


@server.get("/api/graph")
def api_graph():
    return {"graph_html": _build_graph()}


@server.post("/api/export")
def api_export(body: ExportBody):
    path = do_export(body.format_type, body.history)
    if path is None:
        return {"filename": None, "content": None}
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    with contextlib.suppress(OSError):
        os.unlink(path)
    return {"filename": path.name, "content": content}


if __name__ == "__main__":
    restore_index()
    uvicorn.run("server:server", host="0.0.0.0", port=7860, log_level="info")
