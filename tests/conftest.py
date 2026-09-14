import math

import pytest
from langchain_core.documents import Document


class FakeEmbeddings:
    """Deterministic, offline embeddings for store/search tests."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self._terms = ["login", "auth", "users", "token", "api", "config", "test", "main"]

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        lower = text.lower()
        for i, term in enumerate(self._terms):
            if term in lower:
                v[i] = 1.0 + min(len(text) / 10000.0, 0.5)
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="import os\nimport sys\n\ndef main():\n    print('hello')\n",
            metadata={"source": "app.py", "repo": "test-repo", "repo_url": "https://github.com/test/repo"},
        ),
        Document(
            page_content="class Foo:\n    def bar(self):\n        pass\n",
            metadata={"source": "utils.py", "repo": "test-repo", "repo_url": "https://github.com/test/repo"},
        ),
        Document(
            page_content="# README\nThis is a test project.\n",
            metadata={"source": "README.md", "repo": "test-repo", "repo_url": "https://github.com/test/repo"},
        ),
    ]


@pytest.fixture
def sample_code_doc():
    return Document(
        page_content="""import os
import json

class UserManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_user(self, user_id: int) -> dict:
        with open(self.db_path) as f:
            data = json.load(f)
        return data.get(user_id, {})

def create_admin():
    return UserManager("/tmp/admin.db")
""",
        metadata={"source": "users.py", "repo": "test-repo", "repo_url": "https://github.com/test/repo"},
    )
