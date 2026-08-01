import logging
import os
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain.agents import AgentExecutor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


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
        "MAX_HISTORY_TURNS": os.getenv("MAX_HISTORY_TURNS", "20"),
    }


def _log_effective_config() -> None:
    logger.info("Effective configuration:")
    for key, value in _effective_config().items():
        logger.info("  %s=%s", key, value)


_log_effective_config()

import app_theme  # noqa: E402
from rag import (  # noqa: E402
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
    render_graph_html,
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
    documents: list | None = None
    llm: object | None = None


state = AppState()


def _format_chat_history(history: list[dict], max_turns: int = 20, summary: str = "") -> str:
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
    max_turns = int(os.getenv("MAX_HISTORY_TURNS", "20"))
    if len(history) <= max_turns * 2 or state.llm is None:
        return summary
    old = history[: -(max_turns * 2)]
    text = "\n".join(f"{'Human' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" for msg in old)
    return _summarize_conversation(state.llm, summary, text)


def index_repo(github_url: str) -> tuple[str, str]:
    if not github_url.strip():
        return "Please enter a GitHub URL.", ""

    try:
        validate_github_url(github_url)
    except ValueError as e:
        return str(e), ""

    try:
        docs, stats = clone_and_parse(github_url)
        state.documents = docs

        state.embeddings = get_embeddings()
        chunk_count = build_store(state.documents, state.embeddings)

        state.llm = get_llm()

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
                state.agent_executor = AgentExecutor(agent=agent, tools=agent.tools, verbose=False, max_iterations=5)
            except Exception as e:
                logger.warning("Agent initialization failed: %s", e)
                state.agent_executor = None

        repo_name = github_url.strip().rstrip("/").split("/")[-1]
        state.indexed_repo = repo_name

        lang_str = ", ".join(
            f"{ext}: {count}" for ext, count in sorted(stats["files_by_ext"].items(), key=lambda x: -x[1])
        )

        health: list[str] = []
        health.append(f"{'✅' if stats['has_readme'] else '⚠️'} README: {'Found' if stats['has_readme'] else 'Missing'}")
        health.append(f"{'✅' if stats['has_tests'] else '⚠️'} Tests: {'Found' if stats['has_tests'] else 'Missing'}")
        health.append(f"{'✅' if stats['has_ci'] else '⚠️'} CI/CD: {'Found' if stats['has_ci'] else 'Missing'}")
        if stats["todo_count"] > 0:
            health.append(f"📝 TODOs: {stats['todo_count']} found")

        status_text = (
            f"Indexed successfully!\n\n"
            f"Repository: {repo_name}\n"
            f"Files: {stats['total_files']} | Lines: {stats['total_lines']}\n"
            f"Chunks: {chunk_count}\n"
            f"Languages: {lang_str}\n\n" + "\n".join(health) + summary_section + "\n\n"
            f"Provider: {os.getenv('LLM_PROVIDER', 'ollama')}"
        )

        graph_html = _build_graph()

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


def clear_index() -> tuple[str, str, str]:
    state.chain = None
    state.agent_executor = None
    state.indexed_repo = None
    state.documents = None
    state.llm = None
    state.repo_overview = None
    state.file_tree = None
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


with gr.Blocks(
    title="CodeBase QA",
    theme=app_theme.theme,
    css=Path(__file__).resolve().with_name("app.css").read_text(encoding="utf-8"),
) as demo:
    gr.HTML(
        """<div style="display:flex;align-items:center;gap:14px">
  <div class="mark">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 17 10 11 4 5"></polyline><line x1="12" y1="19" x2="20" y2="19"></line></svg>
  </div>
  <div>
    <div class="title">CodeBase QA</div>
    <div class="subtitle">RAG over any GitHub repository &middot; Quick / Deep modes</div>
  </div>
</div>""",
        elem_id="cb-header",
    )

    chat_history = gr.State([])
    chat_summary = gr.State("")

    with gr.Row(elem_id="app-body"):
        with gr.Column(scale=1, elem_id="sidebar"):
            with gr.Accordion("Repository", open=True, elem_id="cb-repo-accordion"):
                github_url = gr.Textbox(
                    label="GitHub Repository URL",
                    placeholder="https://github.com/user/repo",
                )
                with gr.Row():
                    index_btn = gr.Button("Index Repository", variant="primary", elem_id="index-btn")
                    clear_btn = gr.Button("Clear Index", elem_id="clear-btn")
                mode_toggle = gr.Radio(
                    choices=["Quick", "Deep Analysis"],
                    value="Quick",
                    label="Analysis Mode",
                    elem_id="mode-toggle",
                )
                status_output = gr.Textbox(
                    label="Status",
                    lines=10,
                    interactive=False,
                    elem_id="status-output",
                )

            with gr.Accordion("Export", open=False, elem_id="cb-export-accordion"):
                with gr.Row():
                    export_format = gr.Dropdown(
                        choices=["Markdown", "Notebook"],
                        value="Markdown",
                        label="Format",
                    )
                    export_btn = gr.Button("Export Chat")
                export_file = gr.File(label="Download", visible=True)

        with gr.Column(scale=2, elem_id="main"), gr.Tabs(elem_id="cb-tabs"):
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    type="messages",
                    show_copy_button=True,
                    elem_id="cb-chatbot",
                )
                with gr.Row(elem_id="composer"):
                    msg_input = gr.Textbox(
                        label="Ask a question",
                        placeholder="What does this project do?",
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")
                with gr.Row():
                    clear_chat_btn = gr.Button("Clear Chat", elem_id="clear-chat-btn")
                    gr.Examples(
                        examples=[
                            "What does this project do?",
                            "What are the main files?",
                            "Explain the architecture",
                        ],
                        inputs=msg_input,
                        label="Examples",
                        elem_id="cb-examples",
                    )

            with gr.Tab("Dependency Graph"):
                graph_output = gr.HTML(
                    value="<p>Index a repository to view its dependency graph.</p>",
                    label="Dependency Graph",
                    elem_id="cb-graph",
                )

    def user_send(message: str, history: list[dict], mode: str, summary: str):
        if not message.strip():
            yield list(history), "", list(history), summary
            return
        for updated, new_summary in ask_question(message, history, mode, summary):
            yield list(updated), "", list(updated), new_summary

    msg_input.submit(
        fn=user_send,
        inputs=[msg_input, chat_history, mode_toggle, chat_summary],
        outputs=[chatbot, msg_input, chat_history, chat_summary],
    )

    send_btn.click(
        fn=user_send,
        inputs=[msg_input, chat_history, mode_toggle, chat_summary],
        outputs=[chatbot, msg_input, chat_history, chat_summary],
    )

    clear_chat_btn.click(
        fn=lambda: ([], [], ""),
        outputs=[chatbot, chat_history, chat_summary],
    )

    index_btn.click(
        fn=index_repo,
        inputs=[github_url],
        outputs=[status_output, graph_output],
    )

    clear_btn.click(
        fn=clear_index,
        outputs=[status_output, github_url, graph_output],
    )

    export_btn.click(
        fn=do_export,
        inputs=[export_format, chat_history],
        outputs=[export_file],
    )

    gr.HTML(
        """<div class="line">Built by <a href="https://github.com/zakisserious" target="_blank" rel="noopener">zakisserious</a> &middot; LangChain + ChromaDB + Gradio</div>""",
        elem_id="cb-footer",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
