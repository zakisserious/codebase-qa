import json
import logging
import re

from langchain.agents.agent import MultiActionAgentOutputParser
from langchain.agents.output_parsers.tools import (
    ToolAgentAction,
    parse_ai_message_to_tool_action,
)
from langchain_core.agents import AgentFinish
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_REGEX_LENGTH = 200
MAX_SEARCH_RESULTS = 30


class SearchCodeInput(BaseModel):
    pattern: str = Field(description="Regex pattern to search for in code")


class ReadFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to read")
    start_line: int = Field(default=1, description="Start line number")
    end_line: int = Field(default=50, description="End line number")


class GetImportsInput(BaseModel):
    file_path: str = Field(description="Path to the file")


class FindDefinitionsInput(BaseModel):
    name: str = Field(description="Function or class name to find")


class AgentTools:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def _search_code(self, pattern: str) -> str:
        if len(pattern) > MAX_REGEX_LENGTH:
            return f"Pattern too long (max {MAX_REGEX_LENGTH} chars). Use a simpler pattern."

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex pattern: {pattern} ({e})"

        results: list[str] = []
        for doc in self.documents:
            lines = doc.page_content.splitlines()
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    results.append(f"{doc.metadata['source']}:{i}: {line.strip()}")
                    if len(results) >= MAX_SEARCH_RESULTS:
                        return "\n".join(results) + f"\n\n(Showing first {MAX_SEARCH_RESULTS} results)"

        if not results:
            return f"No matches found for '{pattern}'"

        return "\n".join(results)

    def _read_file(self, file_path: str, start_line: int = 1, end_line: int = 50) -> str:
        for doc in self.documents:
            if doc.metadata["source"] == file_path:
                lines = doc.page_content.splitlines()
                selected = lines[start_line - 1 : end_line]
                return "\n".join(f"{start_line + i}: {line}" for i, line in enumerate(selected))
        return f"File not found: {file_path}"

    def _get_file_tree(self) -> str:
        files = sorted(set(doc.metadata["source"] for doc in self.documents))
        return "\n".join(files)

    def _get_imports(self, file_path: str) -> str:
        for doc in self.documents:
            if doc.metadata["source"] == file_path:
                lines = doc.page_content.splitlines()
                import_lines = [line for line in lines if line.strip().startswith(("import ", "from "))]
                return "\n".join(import_lines) if import_lines else "No imports found."
        return f"File not found: {file_path}"

    def _find_definitions(self, name: str) -> str:
        pattern = re.compile(rf"^\s*(?:class|def|async\s+def)\s+{re.escape(name)}\b")
        results: list[str] = []
        for doc in self.documents:
            lines = doc.page_content.splitlines()
            for i, line in enumerate(lines, 1):
                if pattern.match(line):
                    end = min(i + 10, len(lines))
                    preview = "\n".join(lines[i - 1 : end])
                    results.append(f"{doc.metadata['source']}:L{i}-L{end}\n{preview}...")
                    break

        return "\n\n".join(results) if results else f"No definitions found for '{name}'"

    def get_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self._search_code,
                name="search_code",
                description="Search for a regex pattern across all indexed code files. Returns matching lines with file paths and line numbers.",
                args_schema=SearchCodeInput,
            ),
            StructuredTool.from_function(
                func=self._read_file,
                name="read_file",
                description="Read a specific file with optional line range.",
                args_schema=ReadFileInput,
            ),
            StructuredTool.from_function(
                func=self._get_file_tree,
                name="get_file_tree",
                description="Show the directory structure of the indexed repository.",
            ),
            StructuredTool.from_function(
                func=self._get_imports,
                name="get_imports",
                description="Show what a specific file imports.",
                args_schema=GetImportsInput,
            ),
            StructuredTool.from_function(
                func=self._find_definitions,
                name="find_definitions",
                description="Find where a function or class is defined in the codebase.",
                args_schema=FindDefinitionsInput,
            ),
        ]


def _iter_json_blocks(text: str):
    start = -1
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                yield text[start : i + 1]
                start = -1


def _parse_tool_call_jsons(content: str) -> list[dict]:
    text = (content or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    candidates = list(dict.fromkeys([text, *_iter_json_blocks(text)]))

    parsed: list[dict] = []
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("name"), str):
            arguments = data.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if isinstance(arguments, dict):
                data["arguments"] = arguments
                parsed.append(data)
    return parsed


def _format_scratchpad(intermediate_steps: list) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for action, observation in intermediate_steps:
        tool_call_id = getattr(action, "tool_call_id", None) or "call_0"
        messages.append(
            ToolMessage(
                content=f"Result of {action.tool}: {observation}",
                tool_call_id=tool_call_id,
            )
        )
    return messages


class _OllamaToolCallParser(MultiActionAgentOutputParser):
    """Parse model output that uses either native tool calls or JSON-text tool calls.

    Ollama models like qwen2.5-coder often emit tool calls as JSON text
    (``{"name": ..., "arguments": {...}}``) instead of structured ``tool_calls``.
    This parser handles both forms, falling back to treating the output as a final
    answer when neither is present.

    Only tool names that appear in ``valid_tool_names`` are accepted as actions;
    unrecognized names are treated as part of the final answer, preventing
    hallucinated tool calls from crashing the agent.
    """

    _valid_tool_names: set[str] | None = None

    def __init__(self, valid_tool_names: set[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_valid_tool_names", valid_tool_names)

    @property
    def _type(self) -> str:
        return "ollama-tool-call-parser"

    def parse(self, text: str):
        msg = "Can only parse messages"
        raise ValueError(msg)  # noqa: TRY004

    def parse_result(self, result, *, partial=False):
        if not isinstance(result[0], ChatGeneration):
            msg = "This output parser only works on ChatGeneration output"
            raise ValueError(msg)  # noqa: TRY004

        message = result[0].message
        if isinstance(message, AIMessage) and message.tool_calls:
            return parse_ai_message_to_tool_action(message)

        content = getattr(message, "content", "") or ""
        data_list = _parse_tool_call_jsons(content)
        if data_list:
            actions = []
            for data in data_list:
                tool_name = data["name"]
                valid = self._valid_tool_names
                if valid is not None and tool_name not in valid:
                    return AgentFinish(return_values={"output": content}, log=str(content))
                log = f"\nInvoking: `{tool_name}` with `{data['arguments']}`\n"
                actions.append(
                    ToolAgentAction(
                        tool=tool_name,
                        tool_input=data["arguments"],
                        log=log,
                        message_log=[message],
                        tool_call_id="",
                    )
                )
            return actions

        return AgentFinish(return_values={"output": content}, log=str(content))


def build_agent(llm: BaseChatModel, documents: list[Document]):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.tools.render import render_text_description

    tools_instance = AgentTools(documents)
    tools = tools_instance.get_tools()
    tool_names = {t.name for t in tools}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are CodeBase QA, a code analysis agent with access to tools that let you search, read, and analyze code in an indexed repository. You are not a general-purpose model. Never identify yourself as a specific model (such as Qwen, Llama, or GPT) or as its creator; if asked who you are, say you are CodeBase QA.

You have the following tools available:
{tools}

INSTRUCTIONS:
- For each question, use at least one tool to investigate before answering.
- Start by using `get_file_tree` to understand the repository structure, then use `search_code`, `read_file`, `find_definitions`, or `get_imports` to find relevant code.
- Always cite file paths and line numbers in your final answer.
- When you have enough information to answer, respond with a final answer — do not call any more tools.

Previous conversation:
{history}

Use the previous conversation to answer follow-ups; do not claim you lack memory of this chat.""",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ).partial(
        tools=render_text_description(tools),
    )

    llm_with_tools = llm.bind_tools(tools)

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: _format_scratchpad(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_tools
        | _OllamaToolCallParser(valid_tool_names=tool_names)
    )
    object.__setattr__(agent, "tools", tools)
    return agent
