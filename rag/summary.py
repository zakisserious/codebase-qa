import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """Analyze this codebase and provide a brief summary.

File tree:
{file_tree}

Key files (first 50 lines each):
{key_files}

Respond in this exact format (no other text):
SUMMARY: <2-4 sentence description of what this project does>
TECHNOLOGIES: <comma-separated list of frameworks/libraries used>
ENTRY_POINTS: <comma-separated list of main entry point files>"""


def generate_summary(
    documents: list[Document],
    llm: BaseChatModel,
) -> dict:
    logger.info("Generating summary for %d documents...", len(documents))
    files = sorted(set(doc.metadata["source"] for doc in documents))
    file_tree = "\n".join(files)
    key_files = _extract_key_files(documents)

    prompt = SUMMARY_PROMPT.format(file_tree=file_tree, key_files=key_files)

    try:
        response = llm.invoke(prompt)
        parsed = _parse_response(response.content)
        logger.info("Summary generated successfully")
    except Exception as e:
        logger.warning("Summary generation failed: %s", e)
        parsed = {
            "summary": "Unable to generate summary.",
            "technologies": [],
            "entry_points": [],
        }

    return {
        "description": parsed["summary"],
        "technologies": parsed["technologies"],
        "entry_points": parsed["entry_points"],
    }


def _extract_key_files(documents: list[Document], max_files: int = 5) -> str:
    key_names = {
        "app.py",
        "main.py",
        "index.js",
        "index.ts",
        "cli.py",
        "__main__.py",
        "server.py",
        "manage.py",
        "wsgi.py",
        "asgi.py",
        "package.json",
        "requirements.txt",
        "pyproject.toml",
    }
    key_docs: list[str] = []
    for doc in documents:
        name = Path(doc.metadata["source"]).name
        if name in key_names or name.lower().startswith("readme"):
            lines = doc.page_content.splitlines()[:50]
            key_docs.append(f"--- {doc.metadata['source']} ---\n" + "\n".join(lines))
        if len(key_docs) >= max_files:
            break
    return "\n\n".join(key_docs) if key_docs else "No key files identified."


def _parse_response(text: str) -> dict:
    result: dict = {"summary": "", "technologies": [], "entry_points": []}

    summary_match = re.search(r"SUMMARY:\s*(.+?)(?=\nTECHNOLOGIES:|$)", text, re.DOTALL)
    if summary_match:
        result["summary"] = summary_match.group(1).strip()

    tech_match = re.search(r"TECHNOLOGIES:\s*(.+?)(?=\nENTRY_POINTS:|$)", text, re.DOTALL)
    if tech_match:
        raw = tech_match.group(1).strip()
        result["technologies"] = [t.strip() for t in raw.split(",") if t.strip()]

    entry_match = re.search(r"ENTRY_POINTS:\s*(.+?)$", text, re.DOTALL)
    if entry_match:
        raw = entry_match.group(1).strip()
        result["entry_points"] = [e.strip() for e in raw.split(",") if e.strip()]

    return result
