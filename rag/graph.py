import ast
import json
import logging
import posixpath
import re
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "graph.html"


def build_dependency_graph(documents: list[Document]) -> dict:
    logger.info("Building dependency graph for %d documents...", len(documents))
    nodes = []
    node_ids = set()
    links = []
    node_map = {}

    for doc in documents:
        source = doc.metadata["source"]
        ext = Path(source).suffix
        lines = doc.page_content.count("\n") + 1
        ftype = _file_type(source)
        node_map[source] = {"ext": ext, "lines": lines, "type": ftype}
        node_ids.add(source)

    for source in node_ids:
        attrs = node_map[source]
        nodes.append(
            {
                "id": source,
                "name": Path(source).name,
                "ext": attrs["ext"],
                "lines": attrs["lines"],
                "type": attrs["type"],
            }
        )

    doc_by_source = {doc.metadata["source"]: doc for doc in documents}

    for doc in documents:
        source = doc.metadata["source"]
        ext = Path(source).suffix
        imports = _extract_imports(doc.page_content, ext)

        for imp in imports:
            target = _resolve_import(imp, source, doc_by_source)
            if target and target in node_ids and target != source:
                links.append({"source": source, "target": target})

    in_degree = defaultdict(int)
    for link in links:
        in_degree[link["target"]] += 1

    for node in nodes:
        node["in_degree"] = in_degree.get(node["id"], 0)

    most_imported = sorted(nodes, key=lambda n: -n["in_degree"])[:5]
    orphans = [n["id"] for n in nodes if n["in_degree"] == 0 and not any(link["source"] == n["id"] for link in links)]

    return {
        "nodes": nodes,
        "links": links,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(links),
            "orphan_files": orphans,
            "most_imported": [{"name": n["name"], "count": n["in_degree"]} for n in most_imported],
        },
    }


def _extract_imports(content: str, ext: str) -> list[str]:
    if ext == ".py":
        return _extract_python_imports(content)
    elif ext in {".js", ".ts", ".tsx", ".jsx"}:
        return _extract_js_imports(content)
    elif ext == ".rs":
        return _extract_rust_imports(content)
    return []


def _extract_rust_imports(content: str) -> list[str]:
    """Extract Rust module declarations and use statements.

    Handles:
    - mod foo; / pub mod foo;  -> declares child module 'foo'
    - mod foo::bar;            -> nested module
    - use foo::bar;            -> imports path
    - use crate::foo::bar;     -> crate-relative path
    - use self::foo;            -> self-relative
    - use super::foo;          -> parent-relative
    """
    imports = []
    # mod foo; / pub mod foo; / mod foo::bar;
    for m in re.finditer(r"\bmod\s+([a-zA-Z_][a-zA-Z0-9_:://]*)", content):
        path = m.group(1).rstrip(";")
        segs = [s for s in path.split("::") if s]
        if len(segs) > 1:
            imports.append("::".join(segs[:2]))
        else:
            imports.append(segs[0])
    # use foo::bar; / use crate::foo::bar; etc.
    for m in re.finditer(r"\buse\s+([a-zA-Z_][a-zA-Z0-9_:://]*)", content):
        path = m.group(1).rstrip(";")
        # Keep first two segments for relative resolution
        segs = path.split("::")
        if len(segs) > 1:
            imports.append("::".join(segs[:2]))
        else:
            imports.append(path)
    return imports


def _extract_python_imports(content: str) -> list[str]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0:
                if node.module:
                    imports.append("." * node.level + node.module)
                else:
                    for alias in node.names:
                        imports.append("." * node.level + alias.name)
            elif node.module:
                imports.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
    return imports


def _extract_js_imports(content: str) -> list[str]:
    pattern = r'(?:import\s+.*?from\s+[\'"](.+?)[\'"]|require\s*\(\s*[\'"](.+?)[\'"]\s*\))'
    results = []
    for match in re.finditer(pattern, content):
        imp = match.group(1) or match.group(2)
        if imp:
            results.append(imp)
    return results


def _resolve_import(module: str, file_path: str, doc_by_source: dict) -> str | None:
    norm = {source.replace("\\", "/"): source for source in doc_by_source}
    exts = {".py", ".js", ".ts", ".tsx", ".jsx", ".rs"}

    # Rust: mod foo; in src/main.rs resolves to src/foo.rs or src/foo/mod.rs
    if not module.startswith(".") and "::" not in module:
        src_path = file_path.replace("\\", "/")
        parent = str(Path(src_path).parent)
        for candidate in (parent + "/" + module + ".rs", parent + "/" + module + "/mod.rs"):
            if candidate in norm:
                return norm[candidate]

    if module.startswith("."):
        level = len(module) - len(module.lstrip("."))
        rest = module[level:]
        try:
            base = Path(file_path.replace("\\", "/")).parents[level - 1].as_posix()
        except IndexError:
            return None
        parts = [p for p in rest.split("/") if p] if rest.startswith("/") else [p for p in rest.split(".") if p]
        for ext in exts:
            candidate = posixpath.normpath(posixpath.join(base, *parts) + ext)
            if candidate in norm:
                return norm[candidate]
            candidate = posixpath.normpath(posixpath.join(base, *parts, "__init__") + ext)
            if candidate in norm:
                return norm[candidate]
        return None

    # Rust: use crate::audio or use crate::audio::fifo -> src/audio.rs or src/audio/mod.rs
    if "::" in module:
        segments = module.split("::")
        # Strip leading crate/root markers, keep actual module path
        segments = [s for s in segments if s and s not in ("crate", "self", "super")]
        src_path = file_path.replace("\\", "/")
        # Try resolving from src/ directory
        for candidate in ("src/" + "/".join(segments) + ".rs", "src/" + "/".join(segments) + "/mod.rs"):
            if candidate in norm:
                return norm[candidate]
        # Try resolving relative to the file's parent
        parent = str(Path(src_path).parent)
        for candidate in (parent + "/" + "/".join(segments) + ".rs", parent + "/" + "/".join(segments) + "/mod.rs"):
            if candidate in norm:
                return norm[candidate]

    parts = module.split(".")
    for source, norm_source in norm.items():
        source_parts = norm_source.split("/")
        for ext in exts:
            if source_parts[-1] == parts[-1] + ext:
                return source
            if (
                len(parts) > 1
                and len(source_parts) >= len(parts)
                and source_parts[-len(parts) : -1] == parts[:-1]
                and source_parts[-1].startswith(parts[-1])
            ):
                return source

    return None


def _file_type(source: str) -> str:
    ext = Path(source).suffix
    name = Path(source).name.lower()

    if ext == ".py":
        return "test" if "test" in name else "python"
    if ext == ".rs":
        return "test" if "test" in name else "rust"
    if ext in {".js", ".ts", ".tsx", ".jsx"}:
        return "javascript"
    if ext in {".md", ".txt", ".rst"}:
        return "docs"
    if ext in {".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}:
        return "config"
    return "other"


def render_graph_html(graph_data: dict) -> str:
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    data_json = json.dumps(graph_data).replace("</", "<\\/")
    inner = template.replace("__GRAPH_DATA__", data_json)
    return (
        '<iframe srcdoc="' + inner.replace('"', "&quot;") + '" '
        'style="width:100%;height:700px;border:0;border-radius:12px;"></iframe>'
    )
