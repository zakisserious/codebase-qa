import pytest
from langchain_core.documents import Document

from rag.repo_parser import _compute_stats, _parse_github_url, validate_github_url


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
