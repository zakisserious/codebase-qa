.PHONY: install test lint format dev clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies in venv
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:  ## Run all tests
	python -m pytest tests/ -v

test-integration:  ## Run integration tests only
	python -m pytest tests/test_integration.py -v

test-unit:  ## Run unit tests only
	python -m pytest tests/ -v --ignore=tests/test_integration.py

lint:  ## Run ruff linter
	ruff check .

format:  ## Run ruff formatter
	ruff format .
	ruff check --fix .

format-check:  ## Check formatting without fixing
	ruff format --check .

dev:  ## Run the app locally
	python server.py

clean:  ## Remove build artifacts
	rm -rf .venv __pycache__ **/__pycache__ .pytest_cache chroma_db dist build
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
