import html as html_module
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from rag.agent import AgentTools
from rag.chain import _format_docs, build_chain
from rag.code_splitter import split_documents
from rag.graph import build_dependency_graph, render_graph_html
from rag.summary import generate_summary
from rag.vectorstore import build_store, clear_store, get_retriever


class FakeChatModel(BaseChatModel):
    responses: list
    last_messages: list | None = None

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.last_messages = messages
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.responses[0]))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for response in self.responses:
            yield ChatGenerationChunk(message=AIMessageChunk(content=response))


@pytest.fixture
def real_documents():
    return [
        Document(
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
        ),
        Document(
            page_content="""from users import UserManager

def login(username: str, password: str) -> bool:
    um = UserManager("/tmp/auth.db")
    user = um.get_user(username)
    return user.get("password") == password

def logout():
    pass
""",
            metadata={"source": "auth.py", "repo": "test-repo", "repo_url": "https://github.com/test/repo"},
        ),
        Document(
            page_content="""# CodeBase QA
A tool for asking questions about code.
""",
            metadata={"source": "README.md", "repo": "test-repo", "repo_url": "https://github.com/test/repo"},
        ),
    ]


@pytest.fixture(autouse=True)
def cleanup_chroma():
    yield
    clear_store()


class TestFullPipeline:
    def test_split_to_store_to_retriever(self, real_documents):
        chunks = split_documents(real_documents, chunk_size=500)
        assert len(chunks) > 0
        assert all(c.metadata.get("start_line") for c in chunks)

        with patch("rag.vectorstore.Chroma.from_documents"):
            count = build_store(chunks, MagicMock(), chunk_size=500)
            assert count == len(chunks)

    def test_retriever_returns_relevant_chunks(self, real_documents):
        split_documents(real_documents, chunk_size=500)
        with patch("rag.vectorstore.Chroma") as mock_chroma:
            mock_vs = MagicMock()
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = [
                Document(page_content="def login():", metadata={"source": "auth.py", "start_line": 4, "end_line": 7}),
            ]
            mock_vs.as_retriever.return_value = mock_retriever
            mock_chroma.return_value = mock_vs

            retriever = get_retriever(k=4, embedding_model=MagicMock())
            results = retriever.invoke("How does login work?")
            assert len(results) > 0
            assert "login" in results[0].page_content

    def test_chain_produces_response(self, real_documents):
        chunks = split_documents(real_documents, chunk_size=500)
        formatted = _format_docs(chunks)
        assert "users.py" in formatted or "auth.py" in formatted

        llm = FakeChatModel(responses=["The login function authenticates users."])

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = chunks[:2]

        chain = build_chain(mock_retriever, llm=llm)
        result = chain.invoke(
            {
                "question": "How does login work?",
                "history": "No previous conversation.",
            }
        )
        assert result is not None
        assert len(result) > 0

    def test_chain_includes_overview_and_file_tree(self, real_documents):
        llm = FakeChatModel(responses=["ok"])
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="def login(): pass", metadata={"source": "auth.py", "start_line": 1, "end_line": 1}),
        ]

        chain = build_chain(
            mock_retriever,
            llm=llm,
            repo_overview="Description: A code QA tool.",
            file_tree="README.md\nauth.py\nusers.py",
        )
        chain.invoke({"question": "What does this project do?", "history": "None"})

        system = llm.last_messages[0].content
        assert "A code QA tool." in system
        assert "users.py" in system
        assert "README.md" in system

    def test_streaming_works(self, real_documents):
        chunks = split_documents(real_documents, chunk_size=500)

        llm = FakeChatModel(responses=["chunk1", "chunk2", "chunk3"])

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = chunks[:2]

        chain = build_chain(mock_retriever, llm=llm)
        chunks_received = []
        for chunk in chain.stream(
            {
                "question": "What is this?",
                "history": "",
            }
        ):
            chunks_received.append(chunk)

        assert len(chunks_received) == 3
        assert "".join(chunks_received) == "chunk1chunk2chunk3"


class TestAgentToolsIntegration:
    def test_agent_tools_search_across_files(self, real_documents):
        tools = AgentTools(real_documents)
        result = tools._search_code("UserManager")
        assert "users.py" in result

    def test_agent_tools_file_tree(self, real_documents):
        tools = AgentTools(real_documents)
        tree = tools._get_file_tree()
        assert "auth.py" in tree
        assert "users.py" in tree
        assert "README.md" in tree

    def test_agent_tools_definitions(self, real_documents):
        tools = AgentTools(real_documents)
        result = tools._find_definitions("UserManager")
        assert "users.py" in result

    def test_agent_tools_imports(self, real_documents):
        tools = AgentTools(real_documents)
        result = tools._get_imports("auth.py")
        assert "from users import UserManager" in result


class TestSummaryPipeline:
    def test_summary_with_mock_llm(self, real_documents):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "SUMMARY: A user management and authentication system.\n"
            "TECHNOLOGIES: Python, JSON\n"
            "ENTRY_POINTS: auth.py, users.py"
        )
        result = generate_summary(real_documents, mock_llm)
        assert "user management" in result["description"].lower()
        assert "Python" in result["technologies"]
        assert "auth.py" in result["entry_points"]


class TestGraphPipeline:
    def test_graph_from_real_documents(self, real_documents):
        graph = build_dependency_graph(real_documents)
        assert len(graph["nodes"]) > 0
        assert graph["stats"]["total_nodes"] > 0

        html = render_graph_html(graph)
        assert '<iframe srcdoc="' in html
        inner = html_module.unescape(html)
        assert "<!DOCTYPE html>" in inner
        assert "d3.v7" in inner
