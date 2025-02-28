.PHONY: check-poetry install test lint format help system-deps coverage coverage-html

# Extract target descriptions from comments
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

POETRY := $(shell command -v poetry 2> /dev/null)
UNAME := $(shell uname -s)

system-deps: ## Install system dependencies required for the project
	@echo "Installing system dependencies..."
ifeq ($(UNAME), Darwin)
	@echo "Detected macOS"
	@command -v brew >/dev/null 2>&1 || { echo "Homebrew is required. Please install it."; exit 1; }
	@brew install cmake pkg-config
else ifeq ($(UNAME), Linux)
	@echo "Detected Linux"
	@command -v apt-get >/dev/null 2>&1 && { sudo apt-get update && sudo apt-get install -y cmake pkg-config; } || \
	command -v yum >/dev/null 2>&1 && { sudo yum install -y cmake pkg-config; } || \
	{ echo "Could not determine package manager. Please install cmake and pkg-config manually."; exit 1; }
else
	@echo "Unsupported operating system. Please install cmake and pkg-config manually."
endif

check-poetry: ## Check if Poetry is installed, install if not
ifndef POETRY
	@echo "Poetry is not installed. Installing poetry..."
	@curl -sSL https://install.python-poetry.org | python3 -
	@export PATH="$$HOME/.local/bin:$$PATH"
else
	@echo "Poetry is already installed."
endif

install: check-poetry system-deps ## Install project dependencies
	@echo "Installing dependencies..."
	@poetry install

test: check-poetry ## Run tests
	@echo "Running tests..."
	@poetry run pytest

lint: check-poetry ## Run linters
	@echo "Running linters..."
	@poetry run flake8 babeltron
	@poetry run isort --check babeltron
	@poetry run black --check babeltron

format: check-poetry ## Format code
	@echo "Formatting code..."
	@poetry run isort babeltron
	@poetry run black babeltron

coverage: check-poetry ## Run tests with coverage report
	@echo "Running tests with coverage..."
	@poetry run pytest --cov=babeltron --cov-report=term-missing

coverage-html: check-poetry ## Generate HTML coverage report
	@echo "Generating HTML coverage report..."
	@poetry run pytest --cov=babeltron --cov-report=html
