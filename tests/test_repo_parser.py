import pytest
from langchain_core.documents import Document

from rag.repo_parser import _compute_stats, _parse_github_url, parse_local, validate_github_url


class TestValidateGithubUrl:
    def test_valid_url(self):
        assert validate_github_url("https://github.com/user/repo") == "https://github.com/user/repo"

    def test_strips_trailing_slash(self):
        assert validate_github_url("https://github.com/user/repo/") == "https://github.com/user/repo"

    def test_strips_git_suffix(self):
        url = validate_github_url("https://github.com/user/repo.git")
        assert url == "https://github.com/user/repo"

    def test_www_prefix(self):
        assert validate_github_url("https://www.github.com/user/repo") == "https://github.com/user/repo"

    def test_non_github_raises(self):
        with pytest.raises(ValueError, match="Not a GitHub URL"):
            validate_github_url("https://gitlab.com/user/repo")

    def test_too_few_parts_raises(self):
        with pytest.raises(ValueError, match="Invalid GitHub URL"):
            validate_github_url("https://github.com/user")

    def test_whitespace_handling(self):
        assert validate_github_url("  https://github.com/user/repo  ") == "https://github.com/user/repo"


class TestParseGithubUrl:
    def test_appends_git(self):
        assert _parse_github_url("https://github.com/user/repo") == "https://github.com/user/repo.git"

    def test_strips_existing_git(self):
        assert _parse_github_url("https://github.com/user/repo.git") == "https://github.com/user/repo.git"


class TestComputeStats:
    def test_basic_stats(self, sample_documents):
        stats = _compute_stats(sample_documents)
        assert stats["total_files"] == 3
        assert stats["total_lines"] > 0
        assert stats["has_readme"] is True
        assert ".py" in stats["files_by_ext"]
        assert ".md" in stats["files_by_ext"]

    def test_todo_detection(self):
        docs = [
            Document(
                page_content="# TODO: fix this\n# FIXME: broken\n",
                metadata={"source": "code.py"},
            )
        ]
        stats = _compute_stats(docs)
        assert stats["todo_count"] == 2

    def test_tests_detection(self):
        docs = [
            Document(
                page_content="def test_foo(): pass\n",
                metadata={"source": "tests/test_foo.py"},
            )
        ]
        stats = _compute_stats(docs)
        assert stats["has_tests"] is True

    def test_ci_detection(self):
        docs = [
            Document(
                page_content="name: CI\n",
                metadata={"source": ".github/workflows/ci.yml"},
            )
        ]
        stats = _compute_stats(docs)
        assert stats["has_ci"] is True


class TestParseLocal:
    def test_basic_local(self, tmp_path):
        (tmp_path / "app.py").write_text("import os\n\ndef main():\n    pass\n")
        (tmp_path / "README.md").write_text("# Hello\n")

        docs, stats = parse_local(str(tmp_path))

        sources = {doc.metadata["source"] for doc in docs}
        assert sources == {"app.py", "README.md"}
        assert stats["total_files"] == 2
        assert stats["total_lines"] > 0

    def test_respects_skip_dirs(self, tmp_path):
        (tmp_path / "app.py").write_text("import os\n")
        nested = tmp_path / "node_modules"
        nested.mkdir()
        (nested / "lib.js").write_text("export const x = 1;\n")

        docs, _ = parse_local(str(tmp_path))

        sources = {doc.metadata["source"] for doc in docs}
        assert "app.py" in sources
        assert not any("node_modules" in source for source in sources)

    def test_filters_unsupported_extensions(self, tmp_path):
        (tmp_path / "app.py").write_text("import os\n")
        (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        docs, _ = parse_local(str(tmp_path))

        sources = {doc.metadata["source"] for doc in docs}
        assert sources == {"app.py"}

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Not a directory"):
            parse_local(str(tmp_path / "does-not-exist"))

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No supported files found"):
            parse_local(str(tmp_path))

    def test_repo_metadata(self, tmp_path):
        (tmp_path / "app.py").write_text("import os\n")

        docs, _ = parse_local(str(tmp_path))

        assert docs[0].metadata["repo"] == tmp_path.name
        assert docs[0].metadata["repo_url"] == str(tmp_path.resolve())
