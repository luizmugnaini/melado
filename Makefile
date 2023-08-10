export SHELL := /bin/bash

.PHONY: tests docs

tests:
	poetry run pytest tests

lint:
	poetry run ruff melado/ tests/

typecheck:
	poetry run pyre

docs:
	poetry run sphinx-build docs docs/_build
