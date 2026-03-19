.PHONY: dev test lint

dev:
	pip install -U pip
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check .
