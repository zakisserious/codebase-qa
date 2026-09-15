import ast
import logging
import os
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_AST = {".py"}

EXT_LANG = {
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".sh": "bash",
    ".bash": "bash",
    ".lua": "lua",
    ".r": "r",
    ".dart": "dart",
    ".scala": "scala",
}

TS_QUERIES = {
    "javascript": "(function_declaration) @function (class_declaration) @class (import_statement) @imports",
    "typescript": "(function_declaration) @function (class_declaration) @class (import_statement) @imports (interface_declaration) @interface (type_alias_declaration) @type",
    "tsx": "(function_declaration) @function (class_declaration) @class (import_statement) @imports (interface_declaration) @interface",
    "rust": "(function_item) @function (struct_item) @struct (enum_item) @enum (trait_item) @trait (impl_item) @impl (mod_item) @module (use_declaration) @imports",
    "go": "(function_declaration) @function (method_declaration) @method (type_declaration) @type (import_declaration) @imports",
    "java": "(class_declaration) @class (interface_declaration) @interface (method_declaration) @method (import_declaration) @imports",
    "c": "(function_definition) @function (struct_specifier) @struct (enum_specifier) @enum (preproc_include) @imports",
    "cpp": "(function_definition) @function (class_specifier) @class (struct_specifier) @struct (namespace_definition) @module (preproc_include) @imports",
    "c_sharp": "(class_declaration) @class (struct_declaration) @struct (interface_declaration) @interface (method_declaration) @method (namespace_declaration) @module (using_directive) @imports",
    "ruby": "(method) @function (class) @class (module) @module",
    "php": "(function_definition) @function (class_declaration) @class (namespace_definition) @module",
    "kotlin": "(function_declaration) @function (class_declaration) @class (object_declaration) @class (import_header) @imports",
    "bash": "(function_definition) @function",
    "lua": "(function_definition_statement) @function",
    "r": "(function_definition) @function",
    "scala": "(object_definition) @object (class_definition) @class (trait_definition) @interface (function_definition) @function",
}


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    if chunk_size is None:
        chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    if chunk_overlap is None:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))

    logger.info("Splitting %d documents (chunk_size=%d, overlap=%d)", len(documents), chunk_size, chunk_overlap)

    chunks: list[Document] = []

    for doc in documents:
        source = doc.metadata.get("source", "")
        ext = Path(source).suffix

        if ext in SUPPORTED_AST:
            chunks.extend(_split_python(doc, chunk_size))
        elif ext in EXT_LANG:
            split = _split_tree_sitter(doc, chunk_size, ext)
            chunks.extend(split or _split_lines(doc, chunk_size))
        else:
            chunks.extend(_split_lines(doc, chunk_size))

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


def _split_lines(doc: Document, chunk_size: int) -> list[Document]:
    """Line-accurate fallback: chunks keep their real line positions in the file."""
    chunks: list[Document] = []
    for sub in _chunk_lines(doc.page_content.splitlines(), chunk_size, 1):
        chunks.append(
            Document(
                page_content=sub["text"],
                metadata={
                    **doc.metadata,
                    "node_type": "text",
                    "start_line": sub["start"],
                    "end_line": sub["end"],
                },
            )
        )
    return chunks


def _split_tree_sitter(doc: Document, chunk_size: int, ext: str) -> list[Document]:
    lang_name = EXT_LANG.get(ext)
    query_nodes = TS_QUERIES.get(lang_name or "")
    if lang_name is None or query_nodes is None:
        return []

    parser, language = _load_parser(lang_name)
    if parser is None:
        logger.debug("tree-sitter not available for %s", lang_name)
        return []

    try:
        tree = parser.parse(doc.page_content.encode("utf-8"))
        query = language.query(query_nodes)
        raw = query.captures(tree.root_node)
    except Exception as e:
        logger.debug("tree-sitter query failed for %s: %s", doc.metadata.get("source"), e)
        return []

    captures: dict[str, list] = {}
    if isinstance(raw, dict):
        captures = {name: nodes for name, nodes in raw.items()}
    else:
        for node, name in raw:
            captures.setdefault(name, []).append(node)

    chunks: list[Document] = []
    for name, nodes in captures.items():
        for node in nodes:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            text = node.text.decode("utf-8", errors="replace")

            if len(text) <= chunk_size:
                chunks.append(_ts_chunk(doc, name, text, start_line, end_line, node))
            else:
                sub_lines = doc.page_content.splitlines()[start_line - 1 : end_line]
                for sub in _chunk_lines(sub_lines, chunk_size, start_line):
                    chunks.append(_ts_chunk(doc, f"{name}_part", sub["text"], sub["start"], sub["end"], node))
    return chunks


def _load_parser(lang_name: str) -> tuple:
    """Return (parser, language), working with tree-sitter 0.21 and 0.22."""
    try:
        import tree_sitter_languages

        parser = tree_sitter_languages.get_parser(lang_name)
    except Exception:
        return None, None
    if hasattr(parser, "language"):
        return parser, parser.language
    language = tree_sitter_languages.get_language(lang_name)
    parser.set_language(language)
    return parser, language


def _ts_chunk(doc: Document, name: str, text: str, start_line: int, end_line: int, node) -> Document:
    return Document(
        page_content=text,
        metadata={
            **doc.metadata,
            "node_type": name,
            "start_line": start_line,
            "end_line": end_line,
            "name": _extract_js_name(node),
        },
    )


def _get_ts_query_nodes(lang_name: str) -> str | None:
    return TS_QUERIES.get(lang_name)


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
