.PHONY: check-poetry install test lint format help system-deps coverage coverage-html download-model download-model-small download-model-medium download-model-large serve serve-prod docker-build docker-run docker-compose-up docker-compose-down

# Define model path variable with default value, can be overridden by environment
MODEL_PATH ?= ./models

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
	@poetry run pytest tests/unit

lint: check-poetry ## Run linters
	@echo "Running linters..."
	@poetry run isort --check babeltron
	@poetry run black --check babeltron

format: check-poetry ## Format code
	@echo "Formatting code..."
	@poetry run isort babeltron
	@poetry run black babeltron

coverage: check-poetry ## Run tests with coverage report
	@echo "Running tests with coverage..."
	@poetry run pytest tests/unit --cov=babeltron --cov-report=term

coverage-html: check-poetry ## Generate HTML coverage report
	@echo "Generating HTML coverage report..."
	@poetry run pytest tests/unit --cov=babeltron --cov-report=html
	@echo "HTML coverage report generated in htmlcov/ directory"
	@echo "Open htmlcov/index.html in your browser to view the report"

# Model download commands
download-model: download-model-small ## Download the default (small) translation model

download-model-small: check-poetry ## Download small translation model (418M parameters, ~1GB)
	@echo "Downloading small translation model (418M parameters)..."
	@poetry run python -m babeltron.scripts.download_models --size 418M --output-dir $(MODEL_PATH)

download-model-medium: check-poetry ## Download medium translation model (1.2B parameters, ~2.5GB)
	@echo "Downloading medium translation model (1.2B parameters)..."
	@poetry run python -m babeltron.scripts.download_models --size 1.2B --output-dir $(MODEL_PATH)

download-model-large: check-poetry ## Download large translation model (12B parameters, ~24GB)
	@echo "Downloading large translation model (12B parameters)..."
	@poetry run python -m babeltron.scripts.download_models --size 12B --output-dir $(MODEL_PATH)

# Add these commands to your Makefile
serve: check-poetry ## Run the API server locally
	@echo "Starting API server on http://localhost:8000..."
	@poetry run uvicorn babeltron.app.main:app --reload --host 0.0.0.0 --port 8000

serve-prod: check-poetry ## Run the API server in production mode (no reload)
	@echo "Starting API server in production mode on http://localhost:8000..."
	@poetry run uvicorn babeltron.app.main:app --host 0.0.0.0 --port 8000

# Docker commands
docker-build: ## Build Docker image
	@echo "Building Docker image..."
	@docker build -t babeltron:latest .

docker-run: ## Run Docker container with model volume mount
	@echo "Checking for model files..."
	@if [ ! -d "$(MODEL_PATH)" ] || [ -z "$(shell ls -A $(MODEL_PATH) 2>/dev/null)" ]; then \
		echo "No model files found in $(MODEL_PATH) directory."; \
		read -p "Do you want to download the small model now? (y/n) " answer; \
		if [ "$$answer" = "y" ]; then \
			mkdir -p $(MODEL_PATH); \
			echo "Downloading small model..."; \
			poetry run python -m babeltron.scripts.download_models --size 418M --output-dir $(MODEL_PATH); \
		else \
			echo "Model download skipped. Container may not work properly."; \
		fi; \
	fi
	@echo "Running Docker container..."
	@docker run -p 8000:8000 -v $(shell pwd)/$(MODEL_PATH):/models babeltron:latest

docker-up: ## Build and start services with docker-compose
	@echo "Checking for model files..."
	@if [ ! -d "$(MODEL_PATH)" ] || [ -z "$(shell ls -A $(MODEL_PATH) 2>/dev/null)" ]; then \
		echo "No model files found in $(MODEL_PATH) directory."; \
		read -p "Do you want to download the small model now? (y/n) " answer; \
		if [ "$$answer" = "y" ]; then \
			mkdir -p $(MODEL_PATH); \
			echo "Downloading small model..."; \
			poetry run python -m babeltron.scripts.download_models --size 418M --output-dir $(MODEL_PATH); \
		else \
			echo "Model download skipped. Container may not work properly."; \
		fi; \
	fi
	@echo "Building and starting services with docker-compose..."
	@docker-compose up -d --build
	@echo "Services started successfully. API available at http://localhost:8000"
	@echo "API documentation available at http://localhost:8000/docs"

docker-down:
	@echo "Stopping docker-compose services..."
	@docker-compose down
