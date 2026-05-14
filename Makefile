# Makefile — quality & development command runner for contact-center-copilot.
# Primary entry points:  make qa   (run everything)   |   make test   (fast tests)
# Run `make help` for the full list.

.DEFAULT_GOAL := help
SHELL := /bin/bash

PY    := python3
SRC   := src
TESTS := tests
# Unit tests never call the LLM; a dummy key just satisfies config loading.
TEST_ENV := ANTHROPIC_API_KEY=$${ANTHROPIC_API_KEY:-dummy-key-for-unit-tests}

.PHONY: help install format lint pylint type security audit docs deadcode \
        test coverage precommit hooks-update qa clean

help:  ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install dev dependencies and git hooks
	$(PY) -m pip install -e ".[dev]"
	pre-commit install --install-hooks
	@echo "Hooks installed: pre-commit, commit-msg, pre-push."

format:  ## Auto-format and auto-fix the code (ruff)
	ruff check --fix --exit-zero $(SRC) $(TESTS)
	ruff format $(SRC) $(TESTS)

lint:  ## Lint without modifying files (ruff)
	ruff check $(SRC) $(TESTS)
	ruff format --check $(SRC) $(TESTS)

pylint:  ## Deep static analysis (pylint, fail-under 9.5)
	pylint --rcfile=.pylintrc $(SRC)

type:  ## Static type checking (mypy)
	mypy $(SRC)

security:  ## Security scan (bandit)
	bandit -c pyproject.toml -r $(SRC)

# `audit` ignores PYSEC-2022-42969: ReDoS in the unmaintained transitive `py`
# package (SVN-only code path, unreachable here; no fixed release exists).
audit:  ## Dependency vulnerability audit (pip-audit)
	pip-audit --skip-editable --ignore-vuln PYSEC-2022-42969

docs:  ## Docstring coverage (interrogate)
	interrogate -c pyproject.toml $(SRC)

deadcode:  ## Dead-code detection (vulture)
	vulture

test:  ## Fast unit tests — no coverage gate, no LLM calls
	$(TEST_ENV) pytest -q $(TESTS) --ignore=$(TESTS)/integration

coverage:  ## Unit tests with the 90% coverage gate
	$(TEST_ENV) pytest --cov=$(SRC) --cov-report=term-missing --cov-report=html \
		--cov-fail-under=90 $(TESTS) --ignore=$(TESTS)/integration

precommit:  ## Run every git hook against all files
	pre-commit run --all-files --hook-stage pre-commit
	pre-commit run --all-files --hook-stage pre-push

hooks-update:  ## Bump pinned hook versions to the latest releases
	pre-commit autoupdate

qa:  ## Run all checks (lint, types, security, tests). 90% gate: make coverage
	@fail=0; \
	for step in lint pylint type security audit docs deadcode test; do \
		echo ""; echo "──────── make $$step ────────"; \
		$(MAKE) --no-print-directory $$step || fail=1; \
	done; \
	echo ""; \
	if [ $$fail -ne 0 ]; then \
		echo "QA: some checks FAILED."; exit 1; \
	else \
		echo "QA: all checks passed."; \
	fi

clean:  ## Remove caches and build artifacts
	rm -rf .ruff_cache .mypy_cache .pytest_cache htmlcov .coverage \
		build dist ./*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
