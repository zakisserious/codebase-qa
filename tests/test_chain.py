import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from rag.chain import _format_docs, _truncate_file_tree, build_chain, get_llm


class TestGetLLM:
    def test_ollama_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}), patch("rag.chain.ChatOllama") as mock_cls:
            llm = get_llm()
            mock_cls.assert_called_once()
            assert llm == mock_cls.return_value

    def test_huggingface_provider(self):
        with (
            patch.dict(os.environ, {"LLM_PROVIDER": "huggingface"}),
            patch("rag.chain.HuggingFacePipeline") as mock_pipe,
            patch("rag.chain.ChatHuggingFace") as mock_cls,
        ):
            llm = get_llm()
            mock_pipe.from_model_id.assert_called_once()
            mock_cls.assert_called_once()
            assert llm == mock_cls.return_value

    def test_unknown_provider_raises(self):
        with (
            patch.dict(os.environ, {"LLM_PROVIDER": "unknown"}),
            pytest.raises(ValueError, match="Unknown LLM provider"),
        ):
            get_llm()


class TestFormatDocs:
    def test_format_basic(self):
        docs = [
            Document(
                page_content="print('hello')",
                metadata={"source": "app.py", "start_line": 1, "end_line": 1},
            )
        ]
        result = _format_docs(docs)
        assert "File: app.py (L1-L1)" in result
        assert "print('hello')" in result

    def test_format_with_name(self):
        docs = [
            Document(
                page_content="def foo(): pass",
                metadata={
                    "source": "utils.py",
                    "start_line": 1,
                    "end_line": 1,
                    "node_type": "FunctionDef",
                    "name": "foo",
                },
            )
        ]
        result = _format_docs(docs)
        assert "FunctionDef: foo" in result


class TestBuildChain:
    def test_build_chain_returns_chain(self):
        mock_retriever = MagicMock()
        with patch("rag.chain.get_llm") as mock_llm:
            chain = build_chain(mock_retriever, llm=mock_llm.return_value)
            assert chain is not None

    def test_build_chain_uses_provided_llm(self):
        mock_retriever = MagicMock()
        mock_llm = MagicMock()
        with patch("rag.chain.get_llm") as mock_get_llm:
            build_chain(mock_retriever, llm=mock_llm)
            mock_get_llm.assert_not_called()

    def test_retriever_receives_question_only(self):
        received = {}

        def fake_retrieve(query):
            received["query"] = query
            return [Document(page_content="def login(): pass", metadata={"source": "auth.py"})]

        class FakeChatModel(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "fake-chat-model"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

        retriever = RunnableLambda(fake_retrieve)
        chain = build_chain(retriever, llm=FakeChatModel())
        chain.invoke({"question": "How does login work?", "history": "None"})
        assert received["query"] == "How does login work?"

    def test_defaults_when_context_not_provided(self):
        class CapturingChatModel(BaseChatModel):
            messages: list | None = None

            @property
            def _llm_type(self) -> str:
                return "fake-chat-model"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
                self.messages = messages
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

        def fake_retrieve(query):
            return [Document(page_content="def login(): pass", metadata={"source": "auth.py"})]

        llm = CapturingChatModel()
        chain = build_chain(RunnableLambda(fake_retrieve), llm=llm)
        chain.invoke({"question": "What does this project do?", "history": "None"})
        system = llm.messages[0].content
        assert "No repository summary available." in system
        assert "(no file list)" in system

    def test_injected_context_appears_in_system_message(self):
        class CapturingChatModel(BaseChatModel):
            messages: list | None = None

            @property
            def _llm_type(self) -> str:
                return "fake-chat-model"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
                self.messages = messages
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

        def fake_retrieve(query):
            return [Document(page_content="def login(): pass", metadata={"source": "auth.py"})]

        llm = CapturingChatModel()
        chain = build_chain(
            RunnableLambda(fake_retrieve),
            llm=llm,
            repo_overview="Description: A code QA tool.",
            file_tree="app.py\nauth.py\nutils.py",
        )
        chain.invoke({"question": "What does this project do?", "history": "None"})
        system = llm.messages[0].content
        assert "A code QA tool." in system
        assert "app.py" in system
        assert "utils.py" in system

    def test_prompt_instructs_using_history(self):
        class CapturingChatModel(BaseChatModel):
            messages: list | None = None

            @property
            def _llm_type(self) -> str:
                return "fake-chat-model"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
                self.messages = messages
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

        def fake_retrieve(query):
            return [Document(page_content="def login(): pass", metadata={"source": "auth.py"})]

        llm = CapturingChatModel()
        chain = build_chain(
            RunnableLambda(fake_retrieve),
            llm=llm,
            repo_overview="Description: A code QA tool.",
            file_tree="app.py\nauth.py",
        )
        chain.invoke(
            {
                "question": "What did we talk about?",
                "history": "[Earlier conversation summary]: user is Finn\nHuman: my name is finn",
            }
        )
        system = llm.messages[0].content
        assert "recount the conversation when asked" in system
        assert "Never claim you cannot see previous messages" in system
        assert "exists but its contents were not retrieved" in system
        assert "[Earlier conversation summary]" in system
        assert "my name is finn" in system


class TestTruncateFileTree:
    def test_short_tree_unchanged(self):
        assert _truncate_file_tree("a.py\nb.py") == "a.py\nb.py"

    def test_empty_tree(self):
        assert _truncate_file_tree("") == "(no file list)"

    def test_large_tree_truncated(self):
        tree = "\n".join(f"file{i}.py" for i in range(1000))
        result = _truncate_file_tree(tree, max_entries=10)
        assert "file9.py" in result
        assert "file50.py" not in result
        assert "990 more files" in result
