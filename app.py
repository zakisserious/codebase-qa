import contextlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import AgentExecutor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

index_progress: dict[str, str] = {"phase": ""}


def _effective_config() -> dict[str, str]:
    """Resolved configuration (env vars + defaults). Mirrors the defaults used
    in rag/ modules and documented in README, so startup output is truthful."""
    return {
        "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "ollama"),
        "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER", "ollama"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "llama3.1"),
        "OLLAMA_EMBED_MODEL": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        "CHUNK_SIZE": os.getenv("CHUNK_SIZE", "1000"),
        "CHUNK_OVERLAP": os.getenv("CHUNK_OVERLAP", "100"),
        "RETRIEVAL_K": os.getenv("RETRIEVAL_K", "4"),
        "CHROMA_DIR": os.getenv("CHROMA_DIR", "./chroma_db"),
        "ENABLE_AGENT": os.getenv("ENABLE_AGENT", "true"),
        "MAX_AGENT_ITERATIONS": os.getenv("MAX_AGENT_ITERATIONS", "15"),
        "MAX_HISTORY_TURNS": os.getenv("MAX_HISTORY_TURNS", "50"),
    }


def _log_effective_config() -> None:
    logger.info("Effective configuration:")
    for key, value in _effective_config().items():
        logger.info("  %s=%s", key, value)


_log_effective_config()

from rag import (  # noqa: E402
    IndexCancelledError,
    build_agent,
    build_chain,
    build_dependency_graph,
    build_store,
    clear_store,
    clone_and_parse,
    export_session,
    generate_summary,
    get_embeddings,
    get_llm,
    get_retriever,
    parse_local,
    render_graph_html,
    search_documents,
    validate_github_url,
)


@dataclass
class AppState:
    chain: object | None = None
    agent_executor: AgentExecutor | None = None
    indexed_repo: str | None = None
    repo_overview: str | None = None
    file_tree: str | None = None
    embeddings: object | None = None
    retriever: object | None = None
    files: list | None = None
    documents: list | None = None
    llm: object | None = None


state = AppState()


def _format_chat_history(history: list[dict], max_turns: int = 50, summary: str = "") -> str:
    max_turns = int(os.getenv("MAX_HISTORY_TURNS", str(max_turns)))
    lines: list[str] = []
    if summary:
        lines.append(f"[Earlier conversation summary]: {summary}")
    recent = history[-(max_turns * 2) :]
    for msg in recent:
        role = "Human" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) if lines else "No previous conversation."


def _summarize_conversation(llm: object, existing: str, text: str) -> str:
    prompt = (
        "Summarize the older part of a coding Q&A conversation so that later turns "
        "can keep context.\n\n"
        f"Existing summary (keep any facts that are still relevant and merge them "
        f"into the new summary):\n{existing}\n\n"
        f"Older messages to fold in:\n{text}\n\n"
        "Write one concise summary (3-5 sentences). Keep key facts such as the "
        "user's name, preferences, and goals. Respond with the summary only."
    )
    try:
        result = llm.invoke(prompt)
        return getattr(result, "content", result).strip()
    except Exception as e:
        logger.warning("Conversation summary failed: %s", e)
        return existing


def _update_conversation_summary(history: list[dict], summary: str = "") -> str:
    max_turns = int(os.getenv("MAX_HISTORY_TURNS", "50"))
    if len(history) <= max_turns * 2 or state.llm is None:
        return summary
    old = history[: -(max_turns * 2)]
    text = "\n".join(f"{'Human' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" for msg in old)
    return _summarize_conversation(state.llm, summary, text)


def _reset_state() -> None:
    state.chain = None
    state.agent_executor = None
    state.indexed_repo = None
    state.repo_overview = None
    state.file_tree = None
    state.embeddings = None
    state.retriever = None
    state.files = None
    state.documents = None
    state.llm = None


def _meta_path() -> Path:
    return Path(os.getenv("CHROMA_DIR", "./chroma_db")) / "index_meta.json"


def _save_meta(source: str) -> None:
    if not (Path(os.getenv("CHROMA_DIR", "./chroma_db")) / "chroma.sqlite3").exists():
        return
    try:
        meta = {
            "source": source,
            "repo_name": state.indexed_repo,
            "overview": state.repo_overview,
            "file_tree": state.file_tree,
            "files": state.files or [],
        }
        _meta_path().parent.mkdir(parents=True, exist_ok=True)
        _meta_path().write_text(json.dumps(meta))
    except Exception as e:
        logger.warning("Index meta save failed: %s", e)


def restore_index() -> bool:
    """Rebuild an in-memory index from a previous run (durable across restarts).

    Search/chat come back via the persisted Chroma collection; for local
    sources the raw documents are re-parsed from disk so the graph, file
    overlay and Deep Analysis work too. Called once at server startup.
    """
    if state.indexed_repo:
        return False
    if not _meta_path().exists():
        return False
    try:
        meta = json.loads(_meta_path().read_text(encoding="utf-8"))
        source = meta.get("source", "")
    except Exception as e:
        logger.warning("Index restore: unreadable meta (%s)", e)
        return False
    try:
        state.indexed_repo = meta.get("repo_name")
        state.files = meta.get("files") or []
        state.repo_overview = meta.get("overview")
        state.file_tree = meta.get("file_tree")
        local = Path(source).expanduser()
        if local.is_dir():
            docs, _stats = parse_local(str(local))
            state.documents = docs
        state.embeddings = get_embeddings()
        state.retriever = get_retriever(
            k=int(os.getenv("RETRIEVAL_K", "4")),
            embedding_model=state.embeddings,
        )
        state.llm = get_llm()
        state.chain = build_chain(
            state.retriever,
            llm=state.llm,
            repo_overview=state.repo_overview,
            file_tree=state.file_tree,
        )
        if os.getenv("ENABLE_AGENT", "true").lower() == "true" and state.documents:
            with contextlib.suppress(Exception):
                agent = build_agent(state.llm, state.documents)
                state.agent_executor = AgentExecutor(
                    agent=agent,
                    tools=agent.tools,
                    verbose=False,
                    max_iterations=int(os.getenv("MAX_AGENT_ITERATIONS", "15")),
                    handle_parsing_errors=True,
                    handle_tool_errors=True,
                )
        logger.info("Restored index for %s (%d files)", state.indexed_repo, len(state.files or []))
        return True
    except Exception as e:
        logger.warning("Index restore failed: %s", e)
        _reset_state()
        return False


def index_repo(source: str) -> tuple[str, str]:
    if not source.strip():
        return "Please enter a GitHub URL or local path.", ""

    index_progress["phase"] = "Reading repository"
    index_progress["cancel"] = False
    try:
        if Path(source).expanduser().is_dir():
            docs, stats = parse_local(source)
        else:
            validate_github_url(source)
            docs, stats = clone_and_parse(source)
    except ValueError as e:
        return str(e), ""

    if index_progress.get("cancel"):
        return "Index cancelled.", ""

    try:
        state.documents = docs

        state.embeddings = get_embeddings()

        def _embedding_progress(done: int, total: int) -> None:
            index_progress["phase"] = f"Embedding {done}/{total} files"

        try:
            chunk_count = build_store(
                state.documents,
                state.embeddings,
                should_cancel=lambda: bool(index_progress.get("cancel")),
                progress_cb=_embedding_progress,
            )
        except IndexCancelledError:
            _reset_state()
            return "Index cancelled.", ""

        state.llm = get_llm()

        index_progress["phase"] = "Summarizing repository"
        summary_section = ""
        try:
            summary = generate_summary(state.documents, state.llm)
            state.repo_overview = (
                f"Description: {summary['description']}\n"
                f"Technologies: {', '.join(summary['technologies'])}\n"
                f"Entry points: {', '.join(summary['entry_points'])}"
            )
            summary_section = f"\n\n{state.repo_overview}"
        except Exception as e:
            logger.warning("Summary generation failed: %s", e)

        retriever = get_retriever(
            k=int(os.getenv("RETRIEVAL_K", "4")),
            embedding_model=state.embeddings,
        )
        state.retriever = retriever
        state.files = sorted(doc.metadata["source"] for doc in state.documents)
        index_progress["phase"] = "Building retriever & chain"
        state.file_tree = "\n".join(sorted(doc.metadata["source"] for doc in state.documents))
        state.chain = build_chain(
            retriever,
            llm=state.llm,
            repo_overview=state.repo_overview,
            file_tree=state.file_tree,
        )

        if os.getenv("ENABLE_AGENT", "true").lower() == "true":
            try:
                agent = build_agent(state.llm, state.documents)
                state.agent_executor = AgentExecutor(
                    agent=agent,
                    tools=agent.tools,
                    verbose=False,
                    max_iterations=int(os.getenv("MAX_AGENT_ITERATIONS", "15")),
                    handle_parsing_errors=True,
                    handle_tool_errors=True,
                )
            except Exception as e:
                logger.warning("Agent initialization failed: %s", e)
                state.agent_executor = None

        repo_name = docs[0].metadata["repo"]
        state.indexed_repo = repo_name
        _save_meta(source)

        lang_str = ", ".join(
            f"{ext}: {count}" for ext, count in sorted(stats["files_by_ext"].items(), key=lambda x: -x[1])
        )

        health: list[str] = []
        health.append(f"README: {'found' if stats['has_readme'] else 'missing'}")
        health.append(f"Tests: {'found' if stats['has_tests'] else 'missing'}")
        health.append(f"CI/CD: {'found' if stats['has_ci'] else 'missing'}")
        if stats["todo_count"] > 0:
            health.append(f"TODOs: {stats['todo_count']} found")

        status_text = (
            f"Indexed successfully!\n\n"
            f"Repository: {repo_name}\n"
            f"Files: {stats['total_files']} | Lines: {stats['total_lines']}\n"
            f"Chunks: {chunk_count}\n"
            f"Languages: {lang_str}\n\n" + "\n".join(health) + summary_section + "\n\n"
            f"Provider: {os.getenv('LLM_PROVIDER', 'ollama')}"
        )

        graph_html = _build_graph()
        index_progress["phase"] = "Ready"

        return status_text, graph_html

    except Exception as e:
        logger.error("Indexing failed: %s", e)
        return f"Error: {str(e)}", ""


def _build_graph() -> str:
    if not state.documents:
        return "<p>Index a repository to view its dependency graph.</p>"
    try:
        graph_data = build_dependency_graph(state.documents)
        return render_graph_html(graph_data)
    except Exception as e:
        logger.warning("Graph build failed: %s", e)
        return f"<p>Error building graph: {e}</p>"


def ask_question(message: str, history: list[dict], mode: str = "Quick", summary: str = ""):
    if state.chain is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Please index a repository first."})
        yield history, summary
        return

    history.append({"role": "user", "content": message})

    if mode == "Deep Analysis" and state.agent_executor is not None:
        history_text = _format_chat_history(history[:-1], summary=summary)
        try:
            result = state.agent_executor.invoke({"input": message, "history": history_text})
            answer = result.get("output", "No response from agent.")
        except Exception as e:
            logger.error("Agent error: %s", e)
            answer = f"Error: {str(e)}"
        history.append({"role": "assistant", "content": answer})
        new_summary = _update_conversation_summary(history, summary)
        yield history, new_summary
    else:
        history_text = _format_chat_history(history[:-1], summary=summary)
        history.append({"role": "assistant", "content": ""})
        try:
            response = ""
            for chunk in state.chain.stream(
                {
                    "question": message,
                    "history": history_text,
                }
            ):
                response += chunk
                history[-1] = {"role": "assistant", "content": response}
                yield list(history), summary
        except Exception as e:
            logger.error("Chain error: %s", e)
            history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        new_summary = _update_conversation_summary(history, summary)
        yield list(history), new_summary


def collect_sources(question: str, k: int = 10) -> list[dict]:
    """Return the best retrieved chunk per file for a question, for UI citations."""
    if not (question or "").strip() or state.embeddings is None:
        return []
    return search_documents(question, k=k, embedding_model=state.embeddings)


def read_file(path: str, max_lines: int = 4000) -> dict:
    if not state.documents:
        return {"error": "No repository indexed."}
    norm = (path or "").replace("\\", "/")
    for doc in state.documents:
        if doc.metadata.get("source", "").replace("\\", "/") == norm:
            lines = doc.page_content.splitlines() or [""]
            total = len(lines)
            truncated = total > max_lines
            return {
                "source": doc.metadata.get("source", norm),
                "total_lines": total,
                "truncated": truncated,
                "lines": lines[:max_lines],
            }
    return {"error": f"File not found: {path}"}


def classify_agent_chunk(chunk: dict) -> tuple[str, str] | None:
    """Map one AgentExecutor.stream chunk to a UI event.

    Returns ("step", tool_name) for tool executions and ("answer", text) for
    the final output; None for chunks worth ignoring.
    """
    if not isinstance(chunk, dict):
        return None
    output = chunk.get("output")
    if output:
        return ("answer", output)
    for action in chunk.get("actions") or []:
        name = getattr(action, "tool", None) or getattr(action, "name", None)
        if name:
            return ("step", name)
    for msg in chunk.get("messages") or []:
        content = getattr(msg, "content", "") or ""
        m = re.search(r"Invoking:\s*`([^`]+)`", content)
        if m:
            return ("step", m.group(1))
        if not getattr(msg, "tool_calls", None) and getattr(msg, "type", "") == "ai":
            return None
    return None


def stream_deep_analysis(message: str, history: list[dict], summary: str = ""):
    """Stream Deep Analysis. Yields ("step"|"answer"|"done", payload).

    Falls back to a single non-streaming invoke if .stream is not reliable.
    """
    history.append({"role": "user", "content": message})
    if state.agent_executor is None:
        history.append({"role": "assistant", "content": "Deep Analysis is not available for this repository."})
        yield ("answer", "Deep Analysis is not available for this repository.")
        yield ("done", (history, summary))
        return
    history_text = _format_chat_history(history[:-1], summary=summary)
    answer: str | None = None
    try:
        streamed = False
        for chunk in state.agent_executor.stream({"input": message, "history": history_text}):
            streamed = True
            event = classify_agent_chunk(chunk)
            if event is None:
                continue
            kind, value = event
            if kind == "step":
                yield ("step", value)
            elif kind == "answer":
                answer = value
        if answer is None:
            if streamed:
                logger.debug("Agent stream produced no output; falling back to invoke")
            result = state.agent_executor.invoke({"input": message, "history": history_text})
            answer = result.get("output", "No response from agent.")
        if not answer:
            answer = "No response from agent."
    except Exception as e:
        logger.error("Agent error: %s", e)
        answer = f"Error: {str(e)}"
    history.append({"role": "assistant", "content": answer})
    new_summary = _update_conversation_summary(history, summary)
    yield ("answer", answer)
    yield ("done", (history, new_summary))


def clear_index() -> tuple[str, str, str]:
    index_progress["phase"] = ""
    index_progress["cancel"] = False
    _reset_state()
    clear_store()
    return (
        "Index cleared. Enter a new GitHub URL to index.",
        "",
        "<p>Index a repository to view its dependency graph.</p>",
    )


def do_export(format_type: str, history: list[dict]):
    if not state.indexed_repo or not history:
        return None
    repo_name = state.indexed_repo or "unknown"
    return export_session(history, format_type, repo_name)
