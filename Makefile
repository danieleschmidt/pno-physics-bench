.PHONY: help install install-dev test test-cov lint format type-check clean docs serve-docs build check-all

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install package in development mode"
	@echo "  install-dev  Install with development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting (flake8)"
	@echo "  format       Format code (black, isort)"
	@echo "  type-check   Run type checking (mypy)"
	@echo "  clean        Clean build artifacts"
	@echo "  docs         Build documentation"
	@echo "  serve-docs   Serve documentation locally"
	@echo "  build        Build package"
	@echo "  check-all    Run all quality checks"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/pno_physics_bench --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503

format:
	black src/ tests/ --line-length=88
	isort src/ tests/ --profile black --line-length 88

type-check:
	mypy src/pno_physics_bench --strict --ignore-missing-imports

# Documentation
docs:
	cd docs && make html

serve-docs:
	cd docs/_build/html && python -m http.server 8000

# Build
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

# Quality checks
check-all: lint type-check test-cov
	@echo "All quality checks passed!"

# Research workflow shortcuts
train-example:
	python -m pno_physics_bench.cli train --config configs/navier_stokes.yaml

benchmark-all:
	python -m pno_physics_bench.cli benchmark --all-pdes --output results/

visualize-uncertainty:
	python -m pno_physics_bench.visualization --model checkpoints/pno_model.pth