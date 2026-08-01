import logging
import os
import tempfile
from datetime import datetime

logger = logging.getLogger(__name__)


def export_markdown(chat_history: list[dict], repo_name: str = "") -> str:
    md = "# CodeBase QA Session\n\n"
    md += f"**Repository:** {repo_name}\n"
    md += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n"

    for turn in chat_history:
        if turn["role"] == "user":
            md += f"## Q: {turn['content']}\n\n"
        else:
            md += f"{turn['content']}\n\n---\n\n"

    return md


def export_notebook(chat_history: list[dict], repo_name: str = "") -> str:
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    nb.cells.append(
        nbformat.v4.new_markdown_cell(
            f"# CodeBase QA Session\n\n**Repository:** {repo_name}\n"
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
    )

    for turn in chat_history:
        if turn["role"] == "user":
            nb.cells.append(nbformat.v4.new_markdown_cell(f"## Q: {turn['content']}"))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(turn["content"]))

    return nbformat.writes(nb)


def export_session(chat_history: list[dict], format_type: str, repo_name: str = "") -> str:
    if format_type == "Markdown":
        content = export_markdown(chat_history, repo_name)
        suffix = ".md"
    else:
        content = export_notebook(chat_history, repo_name)
        suffix = ".ipynb"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w", prefix="qa_session_")  # noqa: SIM115

    if isinstance(tmp, os.PathLike):
        tmp.write_text(content, encoding="utf-8")
        tmp_name = os.fspath(tmp)
    else:
        with tmp:
            tmp.write(content)
        tmp_name = tmp.name

    logger.info("Exported session to %s", tmp_name)
    return tmp_name
