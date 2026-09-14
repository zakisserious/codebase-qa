import logging
import os
from operator import itemgetter

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

MAX_FILE_TREE_ENTRIES = 500

SYSTEM_PROMPT = """You are a code assistant. Answer the user's question based on the repository overview, file list, and retrieved code snippets below.

Cite your sources using this exact format:
  [filename#L{{start_line}}-{{end_line}}]

Repository overview:
{repo_overview}

All indexed files:
{file_tree}

Retrieved code:
{context}

Grounding rules:
- Answer only from the repository overview, file list, retrieved code above, and the conversation history.
- Use the conversation history to answer follow-ups, remember facts the user stated (such as their name), and recount the conversation when asked. Never claim you cannot see previous messages or that you have no memory of this chat.
- The file list above is the complete set of indexed files, so a listed file exists in the repository even if its content is not among the retrieved code snippets. In that case, say the file exists but its contents were not retrieved, and suggest switching to Deep Analysis.
- Never repeat the conversation history or the "[Earlier conversation summary]" marker in your answer; use the history only to inform your response.
- Never claim a file exists or does not exist unless it is listed in the file list.
- If NO relevant code was retrieved, respond with exactly: "I cannot find specific code for this question in the indexed repository." Then suggest 2-3 candidate files from the file list above, or recommend switching to Deep Analysis mode.

Conversation history (the real previous messages of this chat, in order, including any "[Earlier conversation summary]" line, which summarizes turns older than the window shown above):

{history}"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def get_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    logger.info("Initializing LLM provider: %s", provider)

    if provider == "ollama":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            temperature=0.2,
        )

    if provider == "huggingface":
        pipe = HuggingFacePipeline.from_model_id(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            task="text-generation",
            model_kwargs={"device_map": "auto", "temperature": 0.2},
        )
        return ChatHuggingFace(llm=pipe)

    if provider == "huggingface_api":
        return ChatOpenAI(
            model=os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            openai_api_key=os.getenv("HF_TOKEN"),
            base_url="https://router.huggingface.co/v1",
            temperature=0.2,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")


def _format_docs(docs: list) -> str:
    if not docs:
        return "[No relevant code was retrieved for this question. Base your answer on the file list above and direct the user to candidate files or Deep Analysis mode.]"
    formatted: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        start = doc.metadata.get("start_line", "?")
        end = doc.metadata.get("end_line", "?")
        node_type = doc.metadata.get("node_type", "")
        name = doc.metadata.get("name", "")

        label = f"--- File: {source} (L{start}-L{end})"
        if name:
            label += f" [{node_type}: {name}]"
        label += " ---"

        formatted.append(f"{label}\n{doc.page_content}")
    return "\n\n".join(formatted)


def _truncate_file_tree(file_tree: str, max_entries: int = MAX_FILE_TREE_ENTRIES) -> str:
    lines = [line for line in (file_tree or "").splitlines() if line.strip()]
    if len(lines) <= max_entries:
        return file_tree or "(no file list)"
    return "\n".join(lines[:max_entries]) + f"\n... ({len(lines) - max_entries} more files)"


def build_chain(
    retriever: BaseRetriever,
    llm: BaseChatModel | None = None,
    repo_overview: str | None = None,
    file_tree: str | None = None,
):
    if llm is None:
        llm = get_llm()

    prompt = PROMPT.partial(
        repo_overview=repo_overview or "No repository summary available.",
        file_tree=_truncate_file_tree(file_tree),
    )

    chain = (
        {
            "context": (RunnableLambda(lambda x: x.get("question", "")) | retriever | _format_docs),
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
