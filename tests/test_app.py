import pytest

import app


class FakeChain:
    def __init__(self):
        self.inputs = None

    def stream(self, inputs):
        self.inputs = inputs
        yield "Hello "
        yield "world"


class FakeAgentExecutor:
    def __init__(self):
        self.inputs = None

    def invoke(self, inputs):
        self.inputs = inputs
        return {"output": "deep answer"}


class _AIMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content="user is Finn"):
        self.content = content
        self.prompt = None
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        self.prompt = prompt
        return _AIMessage(self.content)


@pytest.fixture(autouse=True)
def reset_state():
    app.state.chain = None
    app.state.agent_executor = None
    app.state.llm = None
    yield


@pytest.fixture
def clear_history_env(monkeypatch):
    monkeypatch.delenv("MAX_HISTORY_TURNS", raising=False)


def make_history(turns: int) -> list[dict]:
    history = []
    for i in range(turns):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})
    return history


class TestFormatChatHistory:
    def test_empty_history(self):
        assert app._format_chat_history([], max_turns=5) == "No previous conversation."

    def test_roles_and_ordering(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert app._format_chat_history(history, max_turns=5) == "Human: hi\nAssistant: hello"

    def test_truncates_to_max_turns(self, clear_history_env):
        history = make_history(6)
        formatted = app._format_chat_history(history, max_turns=2)
        assert "q0" not in formatted
        assert "q4" in formatted

    def test_includes_rolling_summary(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        formatted = app._format_chat_history(history, max_turns=5, summary="user is Finn")
        assert formatted == ("[Earlier conversation summary]: user is Finn\nHuman: hi\nAssistant: hello")

    def test_summary_only_when_no_turns(self):
        assert app._format_chat_history([], max_turns=5, summary="s") == ("[Earlier conversation summary]: s")

    def test_default_window_keeps_recent_turn(self, clear_history_env):
        formatted = app._format_chat_history(make_history(9))
        assert "q8" in formatted
        assert "q0" not in formatted


class TestAskQuestionMemory:
    def test_quick_mode_keeps_user_and_assistant(self):
        app.state.chain = FakeChain()
        turns = list(app.ask_question("what is login?", [], "Quick"))
        final, summary = turns[-1]
        assert final == [
            {"role": "user", "content": "what is login?"},
            {"role": "assistant", "content": "Hello world"},
        ]
        assert summary == ""

    def test_second_turn_sees_previous_conversation(self):
        chain = FakeChain()
        app.state.chain = chain
        turn1, _ = list(app.ask_question("what is login?", [], "Quick"))[-1]
        list(app.ask_question("where is it used?", turn1, "Quick"))
        assert chain.inputs["history"] == "Human: what is login?\nAssistant: Hello world"

    def test_deep_analysis_passes_history_to_agent(self):
        chain = FakeChain()
        agent = FakeAgentExecutor()
        app.state.chain = chain
        app.state.agent_executor = agent
        turn1, _ = list(app.ask_question("what is login?", [], "Deep Analysis"))[-1]
        assert agent.inputs == {"input": "what is login?", "history": "No previous conversation."}

        list(app.ask_question("where is it used?", turn1, "Deep Analysis"))
        assert "Human: what is login?" in agent.inputs["history"]
        assert "deep answer" in agent.inputs["history"]


class TestEffectiveConfig:
    CONFIG_KEYS = [
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "OLLAMA_MODEL",
        "OLLAMA_EMBED_MODEL",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "RETRIEVAL_K",
        "CHROMA_DIR",
        "ENABLE_AGENT",
        "MAX_AGENT_ITERATIONS",
        "MAX_HISTORY_TURNS",
    ]

    def test_defaults(self, monkeypatch):
        for key in self.CONFIG_KEYS:
            monkeypatch.delenv(key, raising=False)
        assert app._effective_config() == {
            "LLM_PROVIDER": "ollama",
            "EMBEDDING_PROVIDER": "ollama",
            "OLLAMA_MODEL": "llama3.1",
            "OLLAMA_EMBED_MODEL": "nomic-embed-text",
            "CHUNK_SIZE": "1000",
            "CHUNK_OVERLAP": "100",
            "RETRIEVAL_K": "4",
            "CHROMA_DIR": "./chroma_db",
            "ENABLE_AGENT": "true",
            "MAX_AGENT_ITERATIONS": "15",
            "MAX_HISTORY_TURNS": "20",
        }

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "huggingface")
        monkeypatch.setenv("CHUNK_SIZE", "500")
        monkeypatch.setenv("ENABLE_AGENT", "false")
        cfg = app._effective_config()
        assert cfg["LLM_PROVIDER"] == "huggingface"
        assert cfg["CHUNK_SIZE"] == "500"
        assert cfg["ENABLE_AGENT"] == "false"
        assert cfg["MAX_AGENT_ITERATIONS"] == "15"
        assert cfg["MAX_HISTORY_TURNS"] == "20"


class TestRollingSummary:
    def test_summary_created_when_over_window(self, clear_history_env):
        llm = FakeLLM()
        app.state.llm = llm
        history = make_history(21)
        summary = app._update_conversation_summary(history, "")
        assert summary == "user is Finn"
        assert llm.prompt is not None
        assert "Human: q0" in llm.prompt
        assert "Existing summary" in llm.prompt

    def test_summary_carries_existing(self, clear_history_env):
        llm = FakeLLM(content="finn likes FastAPI")
        app.state.llm = llm
        history = make_history(21)
        summary = app._update_conversation_summary(history, "old facts")
        assert summary == "finn likes FastAPI"
        assert "old facts" in llm.prompt

    def test_summary_unchanged_within_window(self, clear_history_env):
        llm = FakeLLM()
        app.state.llm = llm
        history = make_history(20)
        summary = app._update_conversation_summary(history, "existing")
        assert summary == "existing"
        assert llm.calls == 0

    def test_summary_skipped_without_llm(self, clear_history_env):
        history = make_history(21)
        summary = app._update_conversation_summary(history, "existing")
        assert summary == "existing"

    def test_ask_question_folds_old_turns_into_summary(self, clear_history_env):
        app.state.chain = FakeChain()
        app.state.llm = FakeLLM(content="user is Finn")
        history = make_history(21)
        _, summary = list(app.ask_question("q", history, "Quick"))[-1]
        assert summary == "user is Finn"


class TestIndexRepo:
    class _FakeDoc:
        def __init__(self, source="app.py", repo="myrepo"):
            self.page_content = "x = 1\n"
            self.metadata = {"source": source, "repo": repo, "repo_url": "u"}

    @staticmethod
    def _stats():
        return {
            "total_files": 1,
            "total_lines": 1,
            "files_by_ext": {".py": 1},
            "has_readme": True,
            "has_tests": False,
            "has_ci": False,
            "todo_count": 0,
        }

    def _stub_pipeline(self, monkeypatch):
        monkeypatch.setenv("ENABLE_AGENT", "false")
        monkeypatch.setattr(app, "get_embeddings", lambda: object())
        monkeypatch.setattr(app, "build_store", lambda docs, embeddings: 5)
        monkeypatch.setattr(app, "get_llm", lambda: object())
        monkeypatch.setattr(
            app,
            "generate_summary",
            lambda docs, llm: {"description": "d", "technologies": ["python"], "entry_points": ["app.py"]},
        )
        monkeypatch.setattr(app, "get_retriever", lambda k=4, embedding_model=None: object())
        monkeypatch.setattr(app, "build_chain", lambda *args, **kwargs: FakeChain())
        monkeypatch.setattr(app, "_build_graph", lambda: "<html></html>")

    def test_github_url_branch(self, monkeypatch):
        self._stub_pipeline(monkeypatch)
        calls: list = []
        monkeypatch.setattr(app, "clone_and_parse", lambda url: calls.append(url) or ([self._FakeDoc()], self._stats()))

        status, graph = app.index_repo("https://github.com/user/myrepo")

        assert calls == ["https://github.com/user/myrepo"]
        assert status.startswith("Indexed successfully!")
        assert "Repository: myrepo" in status
        assert graph == "<html></html>"

    def test_local_path_branch(self, monkeypatch, tmp_path):
        self._stub_pipeline(monkeypatch)
        (tmp_path / "app.py").write_text("x = 1\n")
        calls: list = []
        monkeypatch.setattr(
            app, "parse_local", lambda path: calls.append(path) or ([self._FakeDoc(repo=tmp_path.name)], self._stats())
        )

        status, _ = app.index_repo(str(tmp_path))

        assert calls == [str(tmp_path)]
        assert status.startswith("Indexed successfully!")

    def test_empty_input(self):
        status, _ = app.index_repo("   ")
        assert status == "Please enter a GitHub URL or local path."
