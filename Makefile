PYTHON ?= python3.11

.PHONY: install test lint format dev clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies in venv
	@test "$$($(PYTHON) -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = "3.11" \
		|| (echo "error: $(PYTHON) is $$($(PYTHON) --version 2>&1) — need Python 3.11; run make install PYTHON=/path/to/python3.11" && exit 1)
	$(PYTHON) -m venv .venv
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
