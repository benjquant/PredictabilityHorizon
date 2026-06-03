.PHONY: help install test lint typecheck format calibration clean

help:
	@echo "make install       - install package + dev deps"
	@echo "make test          - run all tests except calibration"
	@echo "make calibration   - run calibration tests"
	@echo "make lint          - run ruff lint"
	@echo "make typecheck     - run mypy"
	@echo "make format        - run ruff format"
	@echo "make clean         - remove caches"

install:
	uv pip install -e ".[dev]"

test:
	pytest -m "not calibration"

calibration:
	pytest -m calibration

lint:
	ruff check src tests

typecheck:
	mypy

format:
	ruff format src tests

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find src tests -type d -name __pycache__ -exec rm -rf {} +
