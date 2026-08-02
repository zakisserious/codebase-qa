import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import git
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".css",
    ".html",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".next",
    ".venv",
    "venv",
    ".env",
}

MAX_REPO_SIZE_MB = 50


def validate_github_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    parsed = urlparse(url)
    if parsed.hostname not in ("github.com", "www.github.com"):
        raise ValueError(f"Not a GitHub URL: {url}")
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {url}")
    return f"https://github.com/{parts[0]}/{parts[1]}"


def _parse_github_url(url: str) -> str:
    base = validate_github_url(url)
    return base + ".git"


def clone_and_parse(github_url: str) -> tuple[list[Document], dict]:
    repo_url = _parse_github_url(github_url)
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    logger.info("Cloning %s...", repo_url)

    tmp_dir = tempfile.mkdtemp(prefix="codebase_qa_")
    repo_path = os.path.join(tmp_dir, repo_name)

    try:
        git.Repo.clone_from(repo_url, repo_path, depth=1)
        return _read_repo(Path(repo_path), repo_name, github_url, size_cap_mb=MAX_REPO_SIZE_MB)
    except git.exc.GitCommandError as e:
        raise ValueError(f"Failed to clone repository: {e}") from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def parse_local(path: str) -> tuple[list[Document], dict]:
    local_path = Path(path).expanduser().resolve()
    if not local_path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    logger.info("Indexing local directory %s", local_path)
    return _read_repo(local_path, local_path.name, str(local_path), size_cap_mb=None)


def _read_repo(
    repo_path: Path,
    repo_name: str,
    repo_url_meta: str,
    size_cap_mb: int | None = MAX_REPO_SIZE_MB,
) -> tuple[list[Document], dict]:
    if size_cap_mb is not None:
        repo_size = sum(f.stat().st_size for f in repo_path.rglob("*") if f.is_file()) / (1024 * 1024)
        if repo_size > size_cap_mb:
            raise ValueError(f"Repository is {repo_size:.0f}MB, exceeds {size_cap_mb}MB limit.")

    documents = []
    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        if any(skip in file_path.parts for skip in SKIP_DIRS):
            continue
        if file_path.suffix not in SUPPORTED_EXTENSIONS:
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except (UnicodeDecodeError, OSError) as e:
            logger.debug("Skipping %s: %s", file_path, e)
            continue

        if not content.strip():
            continue

        relative = file_path.relative_to(repo_path)
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(relative),
                    "repo": repo_name,
                    "repo_url": repo_url_meta.strip().rstrip("/"),
                },
            )
        )

    if not documents:
        raise ValueError("No supported files found in repository.")

    stats = _compute_stats(documents)
    logger.info("Parsed %d files, %d lines", stats["total_files"], stats["total_lines"])
    return documents, stats


def _compute_stats(documents: list[Document]) -> dict:
    files_by_ext: dict[str, int] = {}
    total_lines = 0
    has_readme = False
    has_tests = False
    has_ci = False
    todo_count = 0
    todo_pattern = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b", re.IGNORECASE)

    ci_indicators = {".github", "workflows", ".gitlab-ci.yml", ".circleci", ".travis.yml"}

    sources_seen: set[str] = set()

    for doc in documents:
        source = doc.metadata["source"]
        ext = Path(source).suffix or "other"
        files_by_ext[ext] = files_by_ext.get(ext, 0) + 1
        total_lines += doc.page_content.count("\n") + 1

        name = Path(source).name.lower()
        if name.startswith("readme"):
            has_readme = True
        if "test" in name or "/tests/" in source or "/__tests__/" in source:
            has_tests = True

        parts = Path(source).parts
        if any(ci in parts for ci in ci_indicators):
            has_ci = True

        todo_count += len(todo_pattern.findall(doc.page_content))

        sources_seen.add(source)

    return {
        "total_files": len(sources_seen),
        "total_lines": total_lines,
        "files_by_ext": files_by_ext,
        "has_readme": has_readme,
        "has_tests": has_tests,
        "has_ci": has_ci,
        "todo_count": todo_count,
    }
