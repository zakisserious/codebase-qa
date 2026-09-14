import json

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
    app.state.documents = None
    app.state.embeddings = None
    app.state.retriever = None
    app.state.files = None
    app.index_progress["cancel"] = False
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
        formatted = app._format_chat_history(make_history(60))
        assert "q59" in formatted
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
            "MAX_HISTORY_TURNS": "50",
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
        assert cfg["MAX_HISTORY_TURNS"] == "50"


class TestRollingSummary:
    def test_summary_created_when_over_window(self, clear_history_env):
        llm = FakeLLM()
        app.state.llm = llm
        history = make_history(101)
        summary = app._update_conversation_summary(history, "")
        assert summary == "user is Finn"
        assert llm.prompt is not None
        assert "Human: q0" in llm.prompt
        assert "Existing summary" in llm.prompt

    def test_summary_carries_existing(self, clear_history_env):
        llm = FakeLLM(content="finn likes FastAPI")
        app.state.llm = llm
        history = make_history(101)
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
        history = make_history(101)
        summary = app._update_conversation_summary(history, "existing")
        assert summary == "existing"

    def test_ask_question_folds_old_turns_into_summary(self, clear_history_env):
        app.state.chain = FakeChain()
        app.state.llm = FakeLLM(content="user is Finn")
        history = make_history(101)
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
        monkeypatch.setattr(app, "build_store", lambda docs, embeddings, **kw: 5)
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


class TestIndexCancel:
    def _stub_doc(self, source, repo):
        return type(
            "D", (), {"page_content": "x = 1\n", "metadata": {"source": source, "repo": repo, "repo_url": "u"}}
        )()

    def _stub_stats(self):
        return {
            "total_files": 1,
            "total_lines": 1,
            "files_by_ext": {".py": 1},
            "has_readme": True,
            "has_tests": False,
            "has_ci": False,
            "todo_count": 0,
        }

    def test_cancel_flag_stops_before_embedding(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ENABLE_AGENT", "false")
        parsed: list = []

        def parse_local(p):
            parsed.append(p)
            app.index_progress["cancel"] = True
            return [self._stub_doc("app.py", tmp_path.name)], self._stub_stats()

        monkeypatch.setattr(app, "parse_local", parse_local)
        status, _ = app.index_repo(str(tmp_path))

        assert status == "Index cancelled."
        assert parsed == [str(tmp_path)]
        assert app.state.files is None

    def test_cancel_in_store_returns_cancelled(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ENABLE_AGENT", "false")
        monkeypatch.setattr(
            app, "parse_local", lambda p: ([self._stub_doc("app.py", tmp_path.name)], self._stub_stats())
        )
        monkeypatch.setattr(app, "get_embeddings", lambda: object())

        from rag.vectorstore import IndexCancelledError

        monkeypatch.setattr(app, "build_store", lambda *a, **kw: (_ for _ in ()).throw(IndexCancelledError("x")))
        status, _ = app.index_repo(str(tmp_path))

        assert status == "Index cancelled."
        assert app.state.files is None


class TestRestoreIndex:
    def test_skip_when_no_meta(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHROMA_DIR", str(tmp_path))
        app._reset_state()
        assert app.restore_index() is False
        assert app.state.indexed_repo is None

    def test_skip_when_already_indexed(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHROMA_DIR", str(tmp_path))
        (tmp_path / "index_meta.json").write_text("{}")
        app._reset_state()
        app.state.indexed_repo = "already-set"
        assert app.restore_index() is False

    def test_restores_state_from_meta(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHROMA_DIR", str(tmp_path))
        monkeypatch.setenv("ENABLE_AGENT", "false")
        monkeypatch.setenv("RETRIEVAL_K", "2")
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("print('hello')\n")
        meta = {
            "source": str(src),
            "repo_name": "demo",
            "overview": "Test overview",
            "file_tree": "main.py",
            "files": ["main.py"],
        }
        (tmp_path / "index_meta.json").write_text(json.dumps(meta))
        stubs = {"embeddings": object(), "llm": object(), "retriever": object()}
        monkeypatch.setattr(app, "get_embeddings", lambda: stubs["embeddings"])
        monkeypatch.setattr(app, "get_llm", lambda: stubs["llm"])
        monkeypatch.setattr(app, "get_retriever", lambda k=4, embedding_model=None: stubs["retriever"])
        monkeypatch.setattr(app, "build_chain", lambda *a, **k: "fake-chain")
        app._reset_state()
        assert app.state.indexed_repo is None
        result = app.restore_index()
        assert result is True
        assert app.state.indexed_repo == "demo"
        assert app.state.files == ["main.py"]
        assert app.state.repo_overview == "Test overview"
        assert app.state.documents is not None and len(app.state.documents) == 1
        assert app.state.retriever is stubs["retriever"]
        assert app.state.chain == "fake-chain"

    def test_index_repo_writes_meta(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHROMA_DIR", str(tmp_path))
        monkeypatch.setenv("ENABLE_AGENT", "false")
        (tmp_path / "chroma.sqlite3").touch()
        fake_doc = type(
            "D", (), {"page_content": "x = 1\n", "metadata": {"source": "main.py", "repo": "myrepo", "repo_url": "u"}}
        )()
        stats = {
            "total_files": 1,
            "total_lines": 1,
            "files_by_ext": {".py": 1},
            "has_readme": True,
            "has_tests": False,
            "has_ci": False,
            "todo_count": 0,
        }
        monkeypatch.setattr(app, "get_embeddings", lambda: object())
        monkeypatch.setattr(app, "get_llm", lambda: object())
        monkeypatch.setattr(app, "build_store", lambda d, e, **kw: 1)
        monkeypatch.setattr(app, "get_retriever", lambda k=4, embedding_model=None: object())
        monkeypatch.setattr(app, "build_chain", lambda *a, **k: object())
        monkeypatch.setattr(app, "_build_graph", lambda: "<html></html>")
        monkeypatch.setattr(app, "parse_local", lambda p: ([fake_doc], stats))
        app._reset_state()
        status, _ = app.index_repo(str(tmp_path))
        assert status.startswith("Indexed successfully!")
        assert (tmp_path / "index_meta.json").exists()
        saved = json.loads((tmp_path / "index_meta.json").read_text())
        assert saved["repo_name"] == "myrepo"
        assert "main.py" in saved["files"]


class TestCollectSources:
    def test_no_index_returns_empty(self):
        app.state.embeddings = None
        assert app.collect_sources("anything") == []

    def test_empty_message_returns_empty(self, monkeypatch):
        called: list = []
        monkeypatch.setattr(app, "search_documents", lambda m, k, embedding_model: called.append((m, k)) or [])
        assert app.collect_sources("   ") == []
        assert called == []


class TestReadFile:
    def _docs(self):
        return [
            {"source": "auth.py", "page_content": "line a\nline b\nline c\n"},
            {"source": "dir/app.py", "page_content": "x\ny\n"},
        ]

    def _fixture(self):
        app.state.documents = [
            type("D", (), {"metadata": d, "page_content": d.pop("page_content")})() for d in self._docs()
        ]

    def test_returns_lines(self):
        self._fixture()
        result = app.read_file("auth.py")
        assert result["total_lines"] == 3
        assert result["lines"] == ["line a", "line b", "line c"]
        assert result["truncated"] is False

    def test_max_lines(self):
        self._fixture()
        result = app.read_file("dir/app.py", max_lines=1)
        assert result["truncated"] is True
        assert result["lines"] == ["x"]

    def test_missing_file(self):
        app.state.documents = []
        result = app.read_file("nope.py")
        assert result["error"]

    def test_no_documents(self):
        result = app.read_file("x.py")
        assert result["error"]


class _FakeAction:
    def __init__(self, tool):
        self.tool = tool


class _Msg:
    def __init__(self, content, type="ai", tool_calls=None):
        self.content = content
        self.type = type
        self.tool_calls = tool_calls or []


class TestClassifyAgentChunk:
    def test_action_step(self):
        event = app.classify_agent_chunk({"actions": [_FakeAction("search_code")]})
        assert event == ("step", "search_code")

    def test_output_answer(self):
        event = app.classify_agent_chunk({"output": "final answer"})
        assert event == ("answer", "final answer")

    def test_invoking_log_step(self):
        event = app.classify_agent_chunk({"messages": [_Msg("Invoking: `read_file` with...")]})
        assert event == ("step", "read_file")

    def test_garbage_is_none(self):
        assert app.classify_agent_chunk(None) is None
        assert app.classify_agent_chunk({"type": "not-a-step"}) is None


class FakeStreamAgent:
    def __init__(self, chunks, invoke_output="fallback answer"):
        self.chunks = chunks
        self.invoke_output = invoke_output
        self.stream_calls = 0
        self.invoke_calls = 0

    def stream(self, inputs):
        self.stream_calls += 1
        yield from self.chunks

    def invoke(self, inputs):
        self.invoke_calls += 1
        return {"output": self.invoke_output}


class TestStreamDeepAnalysis:
    def test_streams_steps_and_answer(self):
        executor = FakeStreamAgent([{"actions": [_FakeAction("search_code")]}, {"output": "deep answer"}])
        app.state.agent_executor = executor
        history = []

        events = [k if isinstance(p, str) else "LIST" for k, p in app.stream_deep_analysis("q", history)]
        assert events == ["step", "answer", "LIST"]
        assert history[-2:] == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "deep answer"},
        ]

    def test_falls_back_to_invoke_when_stream_empty(self):
        executor = FakeStreamAgent([])
        app.state.agent_executor = executor
        history = [{"role": "user", "content": "old"}]

        kinds = []
        for kind, payload in app.stream_deep_analysis("q", history):
            kinds.append(kind)
            if kind == "done":
                done_history, _ = payload

        assert kinds == ["answer", "done"]
        assert executor.stream_calls == 1
        assert executor.invoke_calls == 1
        assert done_history[-1] == {"role": "assistant", "content": "fallback answer"}

    def test_answers_without_executor(self):
        app.state.agent_executor = None
        history = []
        events = []
        for kind, payload in app.stream_deep_analysis("q", history):
            if isinstance(payload, str):
                events.append((kind, payload))
        assert events == [
            ("answer", "Deep Analysis is not available for this repository."),
        ]

    def test_no_tool_calls_yields_no_step(self):
        executor = FakeStreamAgent([{"messages": [_Msg("thinking text")]}], invoke_output="ok")
        app.state.agent_executor = executor
        kinds = [k for k, _ in app.stream_deep_analysis("q", [])]
        assert kinds == ["answer", "done"]
