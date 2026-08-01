import pytest
from langchain_core.documents import Document


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
