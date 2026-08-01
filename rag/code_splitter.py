import ast
import logging
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

SUPPORTED_AST = {".py"}
SUPPORTED_TREE_SITTER = {".js", ".ts", ".tsx", ".jsx"}
FALLBACK_EXTS = {".css", ".html", ".md", ".txt", ".json", ".yaml", ".yml", ".toml"}


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    if chunk_size is None:
        chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    if chunk_overlap is None:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(chunk_size - 1, 0)

    logger.info("Splitting %d documents (chunk_size=%d, overlap=%d)", len(documents), chunk_size, chunk_overlap)

    chunks: list[Document] = []
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    for doc in documents:
        source = doc.metadata.get("source", "")
        ext = Path(source).suffix

        if ext in SUPPORTED_AST:
            chunks.extend(_split_python(doc, chunk_size))
        elif ext in SUPPORTED_TREE_SITTER:
            chunks.extend(_split_js_ts(doc, chunk_size))
        else:
            split = fallback_splitter.split_documents([doc])
            for s in split:
                s.metadata = {
                    **doc.metadata,
                    "node_type": "text",
                    "start_line": 1,
                    "end_line": s.page_content.count("\n") + 1,
                }
            chunks.extend(split)

    logger.info("Produced %d chunks from %d documents", len(chunks), len(documents))
    return chunks


def _split_python(doc: Document, chunk_size: int) -> list[Document]:
    content = doc.page_content
    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.debug("Syntax error parsing %s, falling back to raw text", doc.metadata.get("source"))
        return [
            Document(
                page_content=content,
                metadata={**doc.metadata, "node_type": "text", "start_line": 1, "end_line": content.count("\n") + 1},
            )
        ]

    lines = content.splitlines()
    chunks: list[Document] = []

    imports = [node for node in ast.iter_child_nodes(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

    if imports:
        start = imports[0].lineno
        end = max(getattr(n, "end_lineno", n.lineno) for n in imports)
        import_text = "\n".join(lines[start - 1 : end])
        if import_text.strip():
            chunks.append(
                Document(
                    page_content=import_text,
                    metadata={**doc.metadata, "node_type": "imports", "start_line": start, "end_line": end},
                )
            )

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", node.lineno)
            text = "\n".join(lines[start - 1 : end])

            if len(text) <= chunk_size:
                chunks.append(
                    Document(
                        page_content=text,
                        metadata={
                            **doc.metadata,
                            "node_type": type(node).__name__,
                            "start_line": start,
                            "end_line": end,
                            "name": node.name,
                        },
                    )
                )
            else:
                chunks.extend(_split_large_node(doc, node, lines, chunk_size))

    if not chunks:
        chunks.append(
            Document(
                page_content=content,
                metadata={**doc.metadata, "node_type": "module", "start_line": 1, "end_line": len(lines)},
            )
        )

    return chunks


def _split_large_node(doc: Document, node: ast.AST, lines: list[str], chunk_size: int) -> list[Document]:
    start = node.lineno
    end = getattr(node, "end_lineno", node.lineno)
    name = getattr(node, "name", "unknown")
    node_type = type(node).__name__

    header_lines = lines[start - 1 : start + 1] if start < end else [lines[start - 1]]
    header = "\n".join(header_lines)

    body_start = start + len(header_lines)
    body_lines = lines[body_start - 1 : end]

    chunks: list[Document] = []
    if header.strip():
        chunks.append(
            Document(
                page_content=header,
                metadata={
                    **doc.metadata,
                    "node_type": node_type,
                    "start_line": start,
                    "end_line": min(start + 1, end),
                    "name": name,
                },
            )
        )

    current_chunk: list[str] = []
    current_len = 0
    chunk_start = body_start

    for i, line in enumerate(body_lines):
        line_len = len(line) + 1
        if current_len + line_len > chunk_size and current_chunk:
            chunks.append(
                Document(
                    page_content="\n".join(current_chunk),
                    metadata={
                        **doc.metadata,
                        "node_type": f"{node_type}_body",
                        "start_line": chunk_start,
                        "end_line": body_start + i,
                        "name": name,
                    },
                )
            )
            current_chunk = [line]
            current_len = line_len
            chunk_start = body_start + i
        else:
            current_chunk.append(line)
            current_len += line_len

    if current_chunk:
        chunks.append(
            Document(
                page_content="\n".join(current_chunk),
                metadata={
                    **doc.metadata,
                    "node_type": f"{node_type}_body",
                    "start_line": chunk_start,
                    "end_line": end,
                    "name": name,
                },
            )
        )

    return chunks


def _split_js_ts(doc: Document, chunk_size: int) -> list[Document]:
    try:
        import tree_sitter_languages
    except ImportError:
        logger.debug("tree-sitter not available, falling back to raw text for %s", doc.metadata.get("source"))
        return [
            Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "node_type": "text",
                    "start_line": 1,
                    "end_line": doc.page_content.count("\n") + 1,
                },
            )
        ]

    ext = Path(doc.metadata.get("source", "")).suffix
    lang_map = {".js": "javascript", ".ts": "typescript", ".tsx": "tsx", ".jsx": "jsx"}
    lang_name = lang_map.get(ext, "javascript")

    try:
        parser = tree_sitter_languages.get_parser(lang_name)
    except Exception:
        logger.debug("Could not get tree-sitter parser for %s", lang_name)
        return [
            Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "node_type": "text",
                    "start_line": 1,
                    "end_line": doc.page_content.count("\n") + 1,
                },
            )
        ]

    tree = parser.parse(doc.page_content.encode("utf-8"))
    chunks: list[Document] = []

    query_nodes = _get_ts_query_nodes(lang_name)
    if query_nodes:
        try:
            query = parser.language.query(query_nodes)
            captures = query.captures(tree.root_node)
            for name, nodes in captures.items():
                for node in nodes:
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    text = node.text.decode("utf-8", errors="replace")

                    if len(text) <= chunk_size:
                        chunks.append(
                            Document(
                                page_content=text,
                                metadata={
                                    **doc.metadata,
                                    "node_type": name,
                                    "start_line": start_line,
                                    "end_line": end_line,
                                    "name": _extract_js_name(node),
                                },
                            )
                        )
                    else:
                        sub_lines = doc.page_content.splitlines()[start_line - 1 : end_line]
                        for sub in _chunk_lines(sub_lines, chunk_size, start_line):
                            chunks.append(
                                Document(
                                    page_content=sub["text"],
                                    metadata={
                                        **doc.metadata,
                                        "node_type": f"{name}_part",
                                        "start_line": sub["start"],
                                        "end_line": sub["end"],
                                        "name": _extract_js_name(node),
                                    },
                                )
                            )
        except Exception as e:
            logger.warning("tree-sitter query failed for %s: %s", doc.metadata.get("source"), e)

    if not chunks:
        chunks.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "node_type": "module",
                    "start_line": 1,
                    "end_line": doc.page_content.count("\n") + 1,
                },
            )
        )

    return chunks


def _get_ts_query_nodes(lang_name: str) -> str | None:
    queries: dict[str, str] = {
        "javascript": "(function_declaration) @function (class_declaration) @class (import_statement) @imports",
        "typescript": "(function_declaration) @function (class_declaration) @class (import_statement) @imports (interface_declaration) @interface (type_alias_declaration) @type",
        "tsx": "(function_declaration) @function (class_declaration) @class (import_statement) @imports (interface_declaration) @interface",
        "jsx": "(function_declaration) @function (class_declaration) @class (import_statement) @imports",
    }
    return queries.get(lang_name)


def _extract_js_name(node) -> str:
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8", errors="replace")
    return "anonymous"


def _chunk_lines(lines: list[str], chunk_size: int, base_line: int) -> list[dict]:
    chunks: list[dict] = []
    current: list[str] = []
    current_len = 0
    chunk_start = base_line

    for i, line in enumerate(lines):
        line_len = len(line) + 1
        if current_len + line_len > chunk_size and current:
            chunks.append(
                {
                    "text": "\n".join(current),
                    "start": chunk_start,
                    "end": base_line + i,
                }
            )
            current = [line]
            current_len = line_len
            chunk_start = base_line + i
        else:
            current.append(line)
            current_len += line_len

    if current:
        chunks.append(
            {
                "text": "\n".join(current),
                "start": chunk_start,
                "end": base_line + len(lines),
            }
        )

    return chunks
