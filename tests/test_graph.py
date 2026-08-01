import html as html_module

from langchain_core.documents import Document

from rag.graph import _extract_imports, build_dependency_graph, render_graph_html


def _doc(source: str, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "repo": "test-repo", "repo_url": "https://github.com/test/repo"},
    )


class TestBuildDependencyGraph:
    def test_basic_graph(self, sample_documents):
        graph = build_dependency_graph(sample_documents)
        assert "nodes" in graph
        assert "links" in graph
        assert "stats" in graph
        assert len(graph["nodes"]) > 0

    def test_empty_input(self):
        graph = build_dependency_graph([])
        assert graph["nodes"] == []
        assert graph["links"] == []
        assert graph["stats"]["total_nodes"] == 0

    def test_stats_populated(self, sample_documents):
        graph = build_dependency_graph(sample_documents)
        assert "total_nodes" in graph["stats"]
        assert "total_edges" in graph["stats"]

    def test_absolute_import_links_module(self, sample_documents):
        docs = sample_documents + [_doc("auth.py", "from utils import helper")]
        graph = build_dependency_graph(docs)
        assert {"source": "auth.py", "target": "utils.py"} in graph["links"]

    def test_same_package_relative_import(self):
        docs = [
            _doc("pkg/__init__.py", "from .helper import h"),
            _doc("pkg/helper.py", "def h(): pass"),
        ]
        graph = build_dependency_graph(docs)
        assert {"source": "pkg/__init__.py", "target": "pkg/helper.py"} in graph["links"]

    def test_parent_package_relative_import_without_module(self):
        docs = [
            _doc("pkg/__init__.py", ""),
            _doc("pkg/root_util.py", "def root(): pass"),
            _doc("pkg/sub/__init__.py", ""),
            _doc("pkg/sub/mod.py", "from .. import root_util"),
        ]
        graph = build_dependency_graph(docs)
        assert {"source": "pkg/sub/mod.py", "target": "pkg/root_util.py"} in graph["links"]

    def test_parent_package_relative_import_with_module(self):
        docs = [
            _doc("pkg/__init__.py", ""),
            _doc("pkg/models/__init__.py", ""),
            _doc("pkg/models/user.py", "class User: pass"),
            _doc("pkg/sub/__init__.py", ""),
            _doc("pkg/sub/mod.py", "from ..models import user"),
        ]
        graph = build_dependency_graph(docs)
        assert {"source": "pkg/sub/mod.py", "target": "pkg/models/__init__.py"} in graph["links"]

    def test_multi_level_relative_import(self):
        docs = [
            _doc("top.py", "def top(): pass"),
            _doc("pkg/__init__.py", ""),
            _doc("pkg/sub/__init__.py", ""),
            _doc("pkg/sub/deep.py", "from ...top import x"),
        ]
        graph = build_dependency_graph(docs)
        assert {"source": "pkg/sub/deep.py", "target": "top.py"} in graph["links"]

    def test_js_relative_import(self):
        docs = [
            _doc("src/app.tsx", "import Button from './components/Button';"),
            _doc("src/components/Button.tsx", "export const Button = () => {};"),
            _doc("src/utils.ts", "export const util = 1;"),
            _doc("src/components/toolbar.tsx", "import { util } from '../utils';"),
        ]
        graph = build_dependency_graph(docs)
        assert {"source": "src/app.tsx", "target": "src/components/Button.tsx"} in graph["links"]
        assert {"source": "src/components/toolbar.tsx", "target": "src/utils.ts"} in graph["links"]

    def test_no_self_links(self, sample_documents):
        graph = build_dependency_graph(sample_documents)
        assert all(link["source"] != link["target"] for link in graph["links"])

    def test_extract_relative_imports_keep_levels(self):
        imports = _extract_imports("from .. import root_util\nfrom .helper import h\n", ".py")
        assert "..root_util" in imports
        assert ".helper" in imports


class TestRenderGraphHtml:
    def test_renders_html(self, sample_documents):
        graph = build_dependency_graph(sample_documents)
        html = render_graph_html(graph)
        assert '<iframe srcdoc="' in html
        inner = html_module.unescape(html)
        assert "<!DOCTYPE html>" in inner
        assert "d3.v7" in inner
        assert "nodes" in inner
