.PHONY: check format lint type arch test

check: lint type test

format:
	uv run ruff format .
	uv run ruff check --fix .
lint:
	uv run ruff check .
	uv run ruff format --check .

type:
	uv run mypy

test:
	uv run pytest

arch:
	uv run lint-imports