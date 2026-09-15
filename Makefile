PYTHON ?=

.PHONY: install test lint format dev clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies in venv (auto-detects/fetches Python 3.11)
	@PY="$(PYTHON)"; \
	if [ -z "$$PY" ]; then \
		for c in .venv/bin/python python3.11 python3 python; do \
			if command -v "$$c" >/dev/null 2>&1 && [ "$$($$c -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" = "3.11" ]; then \
				PY="$$c"; break; \
			fi; \
		done; \
	fi; \
	if [ -z "$$PY" ] && command -v uv >/dev/null 2>&1; then \
		echo "no Python 3.11 found — fetching it via uv"; \
		uv venv --python 3.11 .venv; \
		.venv/bin/pip install -r requirements.txt; \
		exit 0; \
	fi; \
	if [ -z "$$PY" ]; then \
		echo "no Python 3.11 found."; \
		echo "Install it via uv (fast, no sudo), then re-run: make install"; \
		echo '  curl -LsSf https://astral.sh/uv/install.sh | sh'; \
		echo "  uv python install 3.11"; \
		echo "Or with your package manager:"; \
		echo "  Debian/Ubuntu: sudo apt install python3.11 python3.11-venv"; \
		echo "  macOS:         brew install python@3.11"; \
		exit 1; \
	fi; \
	echo "using $$PY ($$($$PY --version))"; \
	$$PY -m venv .venv; \
	.venv/bin/pip install -r requirements.txt

test:  ## Run all tests
	.venv/bin/python -m pytest tests/ -v

test-integration:  ## Run integration tests only
	.venv/bin/python -m pytest tests/test_integration.py -v

test-unit:  ## Run unit tests only
	.venv/bin/python -m pytest tests/ -v --ignore=tests/test_integration.py

lint:  ## Run ruff linter
	.venv/bin/ruff check .

format:  ## Run ruff formatter
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

format-check:  ## Check formatting without fixing
	.venv/bin/ruff format --check .

dev:  ## Run the app locally
	.venv/bin/python server.py

clean:  ## Remove build artifacts
	rm -rf .venv __pycache__ **/__pycache__ .pytest_cache chroma_db dist build
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
