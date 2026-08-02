from contextlib import suppress
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from langchain.agents import AgentExecutor
from langchain_core.agents import AgentFinish
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from rag.agent import (
    AgentTools,
    _OllamaToolCallParser,
    _parse_tool_call_jsons,
    build_agent,
)


def generation(text: str) -> list[ChatGeneration]:
    return [ChatGeneration(message=AIMessage(content=text))]


class _FakeAgentLLM(BaseChatModel):
    """Captures rendered messages and supports bind_tools for agent testing."""

    last_messages: ClassVar[list] = []

    @property
    def _llm_type(self) -> str:
        return "fake-agent-llm"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        _FakeAgentLLM.last_messages = list(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def agent_tools():
    docs = [
        Document(
            page_content="def login(): pass\ndef logout(): pass",
            metadata={"source": "auth.py", "name": "login", "start_line": 1, "end_line": 1},
        ),
        Document(
            page_content="import os\nimport sys",
            metadata={"source": "utils.py"},
        ),
    ]
    return AgentTools(docs)


class TestAgentToolsSearch:
    def test_finds_match(self, agent_tools):
        result = agent_tools._search_code("login")
        assert "auth.py" in result
        assert "login" in result

    def test_no_match(self, agent_tools):
        result = agent_tools._search_code("nonexistent")
        assert "No matches" in result

    def test_invalid_regex(self, agent_tools):
        result = agent_tools._search_code("[invalid")
        assert "Invalid regex" in result

    def test_pattern_too_long(self, agent_tools):
        result = agent_tools._search_code("a" * 300)
        assert "too long" in result


class TestAgentToolsReadFile:
    def test_reads_existing(self, agent_tools):
        result = agent_tools._read_file("auth.py", 1, 2)
        assert "login" in result

    def test_file_not_found(self, agent_tools):
        result = agent_tools._read_file("missing.py")
        assert "not found" in result


class TestAgentToolsFileTree:
    def test_returns_files(self, agent_tools):
        result = agent_tools._get_file_tree()
        assert "auth.py" in result
        assert "utils.py" in result


class TestAgentToolsImports:
    def test_finds_imports(self, agent_tools):
        result = agent_tools._get_imports("utils.py")
        assert "import os" in result

    def test_no_imports(self):
        tools = AgentTools([Document(page_content="hello", metadata={"source": "readme.md"})])
        result = tools._get_imports("readme.md")
        assert "No imports" in result


class TestAgentToolsDefinitions:
    def test_finds_definition(self, agent_tools):
        result = agent_tools._find_definitions("login")
        assert "auth.py" in result

    def test_not_found(self, agent_tools):
        result = agent_tools._find_definitions("missing")
        assert "No definitions" in result


class TestGetTools:
    def test_returns_tools(self, agent_tools):
        tools = agent_tools.get_tools()
        assert len(tools) == 5
        names = {t.name for t in tools}
        assert names == {"search_code", "read_file", "get_file_tree", "get_imports", "find_definitions"}


class TestBuildAgent:
    def test_returns_agent_with_tools(self):
        mock_llm = MagicMock()
        docs = [Document(page_content="def foo(): pass", metadata={"source": "a.py"})]
        agent = build_agent(mock_llm, docs)
        assert hasattr(agent, "tools")
        assert len(agent.tools) == 5

    def test_agent_scratchpad_accepts_tool_messages(self):
        mock_llm = MagicMock()
        docs = [Document(page_content="def foo(): pass", metadata={"source": "a.py"})]
        agent = build_agent(mock_llm, docs)
        prompt = agent.steps[1]
        messages = prompt.format_messages(input="hello", history="No previous conversation.", agent_scratchpad=[])
        assert any("hello" in m.content for m in messages)


class TestParseToolCallJsons:
    def test_single(self):
        data = _parse_tool_call_jsons('{"name": "search_code", "arguments": {"pattern": "login"}}')
        assert data == [{"name": "search_code", "arguments": {"pattern": "login"}}]

    def test_multiple_objects(self):
        text = (
            '{"name": "search_code", "arguments": {"pattern": "login"}}\n\n'
            '{"name": "read_file", "arguments": {"file_path": "auth.py"}}'
        )
        data = _parse_tool_call_jsons(text)
        assert len(data) == 2
        assert data[0]["name"] == "search_code"
        assert data[1]["name"] == "read_file"

    def test_arguments_as_json_string(self):
        text = '{"name": "read_file", "arguments": "{\\"file_path\\": \\"auth.py\\"}"}'
        data = _parse_tool_call_jsons(text)
        assert data == [{"name": "read_file", "arguments": {"file_path": "auth.py"}}]

    def test_fenced_block(self):
        data = _parse_tool_call_jsons('```json\n{"name": "search_code", "arguments": {"pattern": "x"}}\n```')
        assert data == [{"name": "search_code", "arguments": {"pattern": "x"}}]

    def test_invalid(self):
        assert _parse_tool_call_jsons("not json at all") == []


class TestOllamaToolCallParser:
    def test_single_json_text_becomes_action(self):
        parser = _OllamaToolCallParser()
        actions = parser.parse_result(generation('{"name": "search_code", "arguments": {"pattern": "login"}}'))
        assert len(actions) == 1
        assert actions[0].tool == "search_code"
        assert actions[0].tool_input == {"pattern": "login"}

    def test_multiple_json_text_becomes_actions(self):
        parser = _OllamaToolCallParser()
        text = (
            '{"name": "search_code", "arguments": {"pattern": "login"}}\n'
            '{"name": "read_file", "arguments": {"file_path": "auth.py"}}'
        )
        actions = parser.parse_result(generation(text))
        assert len(actions) == 2
        assert [a.tool for a in actions] == ["search_code", "read_file"]

    def test_plain_answer_becomes_finish(self):
        parser = _OllamaToolCallParser()
        result = parser.parse_result(generation("The answer is 42"))
        assert isinstance(result, AgentFinish)
        assert result.return_values["output"] == "The answer is 42"

    def test_invalid_tool_name_becomes_finish(self):
        parser = _OllamaToolCallParser(valid_tool_names={"search_code", "read_file"})
        result = parser.parse_result(generation('{"name": "Finn", "arguments": {"foo": "bar"}}'))
        assert isinstance(result, AgentFinish)
        assert result.return_values["output"] == '{"name": "Finn", "arguments": {"foo": "bar"}}'

    def test_valid_tool_name_becomes_action(self):
        parser = _OllamaToolCallParser(valid_tool_names={"search_code", "read_file"})
        actions = parser.parse_result(generation('{"name": "search_code", "arguments": {"pattern": "login"}}'))
        assert len(actions) == 1
        assert actions[0].tool == "search_code"


class TestAgentMemory:
    def test_history_reaches_system_prompt(self):
        mock_llm = _FakeAgentLLM()
        docs = [Document(page_content="def foo(): pass", metadata={"source": "a.py"})]
        agent = build_agent(mock_llm, docs)

        with suppress(Exception):
            agent.invoke({"input": "hi", "history": "MY REAL HISTORY TEXT", "intermediate_steps": []})

        system_msgs = [m for m in _FakeAgentLLM.last_messages if isinstance(m, SystemMessage)]
        assert system_msgs, "No system message was captured by the LLM"
        assert any("MY REAL HISTORY TEXT" in m.content for m in system_msgs), (
            "Conversation history was not injected into the system prompt"
        )
        assert not any("{history}" in m.content for m in system_msgs), (
            "Literal '{history}' placeholder left in system prompt"
        )

    def test_history_reaches_via_agent_executor(self):
        mock_llm = _FakeAgentLLM()
        docs = [Document(page_content="def foo(): pass", metadata={"source": "a.py"})]
        agent = build_agent(mock_llm, docs)

        executor = AgentExecutor(
            agent=agent,
            tools=agent.tools,
            max_iterations=1,
            handle_parsing_errors=True,
            handle_tool_errors=True,
        )

        with suppress(Exception):
            executor.invoke({"input": "test", "history": "AGENT EXECUTOR HISTORY"})

        system_msgs = [m for m in _FakeAgentLLM.last_messages if isinstance(m, SystemMessage)]
        assert system_msgs, "No system message was captured by the LLM"
        assert any("AGENT EXECUTOR HISTORY" in m.content for m in system_msgs), (
            "Conversation history was not injected into the system prompt via AgentExecutor"
        )
