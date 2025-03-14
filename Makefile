.PHONY: check-poetry install test lint format help system-deps coverage coverage-html download-model download-model-m2m-small download-model-m2m-medium download-model-m2m-large download-model-nllb download-model-nllb-small download-model-nllb-medium download-model-nllb-large serve serve-prod docker-build docker-run docker compose-up docker compose-down pre-commit-install pre-commit-run docker-build-with-model docker-up docker-down

# Define model path variable with default value, can be overridden by environment
MODEL_PATH ?= ./models
MODEL_SIZE ?= small
BABELTRON_MODEL_TYPE ?= m2m
IMAGE_NAME ?= babeltron
PORT ?= 8000

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
download-model: download-model-m2m-small ## Download the default (small) translation model

download-model-m2m-small: check-poetry ## Download small translation model (418M parameters, ~1GB)
	@echo "Downloading small translation model (418M parameters)..."
	@poetry run python -m babeltron.scripts.download_models --size 418M --output-dir $(MODEL_PATH)

download-model-m2m-medium: check-poetry ## Download medium translation model (1.2B parameters, ~2.5GB)
	@echo "Downloading medium translation model (1.2B parameters)..."
	@poetry run python -m babeltron.scripts.download_models --size 1.2B --output-dir $(MODEL_PATH)

download-model-m2m-large: check-poetry ## Download large translation model (12B parameters, ~24GB)
	@echo "Downloading large translation model (12B parameters)..."
	@poetry run python -m babeltron.scripts.download_models --size 12B --output-dir $(MODEL_PATH)

# NLLB model download commands
download-model-nllb: download-model-nllb-small ## Download the default (small) NLLB translation model

download-model-nllb-small: check-poetry ## Download small NLLB translation model (600M parameters, ~1.2GB)
	@echo "Downloading small NLLB translation model (600M parameters)..."
	@poetry run python -m babeltron.scripts.download_models --model-type nllb --size 600M --output-dir $(MODEL_PATH)

download-model-nllb-large: check-poetry ## Download large NLLB translation model (3.3B parameters, ~6.6GB)
	@echo "Downloading large NLLB translation model (3.3B parameters)..."
	@poetry run python -m babeltron.scripts.download_models --model-type nllb --size 3.3B --output-dir $(MODEL_PATH)

# Add these commands to your Makefile
serve: check-poetry ## Run the API server in development mode (with reload)
	@echo "Starting API server in development mode on http://localhost:$(PORT)..."
	@poetry run uvicorn babeltron.app.main:app --reload --host 0.0.0.0 --port $(PORT)

serve-prod: check-poetry ## Run the API server in production mode (no reload)
	@echo "Starting API server in production mode on http://localhost:$(PORT)..."
	@poetry run uvicorn babeltron.app.main:app --host 0.0.0.0 --port $(PORT)

# Docker commands
docker-build: ## Build Docker image
	@echo "Building Docker image..."
	@docker build -t $(IMAGE_NAME):latest .

docker-build-with-model: ## Build Docker image with embedded model (use IMAGE_NAME=name to customize image name)
	@echo "Building Docker image with embedded $(BABELTRON_MODEL_TYPE) $(MODEL_SIZE) model..."
	@if [ ! -d "$(MODEL_PATH)" ] || [ -z "$(shell ls -A $(MODEL_PATH) 2>/dev/null)" ]; then \
		echo "No model files found in $(MODEL_PATH) directory. Downloading..."; \
		if [ "$(BABELTRON_MODEL_TYPE)" = "m2m" ]; then \
			if [ "$(MODEL_SIZE)" = "small" ]; then \
				$(MAKE) download-model-m2m-small; \
			elif [ "$(MODEL_SIZE)" = "medium" ]; then \
				$(MAKE) download-model-m2m-medium; \
			elif [ "$(MODEL_SIZE)" = "large" ]; then \
				$(MAKE) download-model-m2m-large; \
			else \
				echo "Invalid model size: $(MODEL_SIZE). Using small model."; \
				$(MAKE) download-model-m2m-small; \
			fi; \
		elif [ "$(BABELTRON_MODEL_TYPE)" = "nllb" ]; then \
			if [ "$(MODEL_SIZE)" = "small" ]; then \
				$(MAKE) download-model-nllb-small; \
			elif [ "$(MODEL_SIZE)" = "large" ]; then \
				$(MAKE) download-model-nllb-large; \
			else \
				echo "Invalid model size: $(MODEL_SIZE). Using small model."; \
				$(MAKE) download-model-nllb-small; \
			fi; \
		else \
			echo "Invalid model type: $(BABELTRON_MODEL_TYPE). Using m2m small model."; \
			$(MAKE) download-model-m2m-small; \
		fi; \
	fi
	@echo "Building Docker image..."
	@docker build -t $(IMAGE_NAME):$(BABELTRON_MODEL_TYPE)-$(MODEL_SIZE) -f Dockerfile.with-model .
	@echo "Docker image with $(BABELTRON_MODEL_TYPE) $(MODEL_SIZE) model built successfully as $(IMAGE_NAME):$(BABELTRON_MODEL_TYPE)-$(MODEL_SIZE)"

docker-run: ## Run Docker container with model volume mount
	@echo "Checking for model files..."
	@if [ ! -d "$(MODEL_PATH)" ] || [ -z "$(shell ls -A $(MODEL_PATH) 2>/dev/null)" ]; then \
		echo "No model files found in $(MODEL_PATH) directory."; \
		read -p "Do you want to download the $(BABELTRON_MODEL_TYPE) $(MODEL_SIZE) model now? (y/n) " answer; \
		if [ "$$answer" = "y" ]; then \
			mkdir -p $(MODEL_PATH); \
			if [ "$(BABELTRON_MODEL_TYPE)" = "m2m" ]; then \
				if [ "$(MODEL_SIZE)" = "small" ]; then \
					$(MAKE) download-model-m2m-small; \
				elif [ "$(MODEL_SIZE)" = "medium" ]; then \
					$(MAKE) download-model-m2m-medium; \
				elif [ "$(MODEL_SIZE)" = "large" ]; then \
					$(MAKE) download-model-m2m-large; \
				else \
					echo "Invalid model size: $(MODEL_SIZE). Using small model."; \
					$(MAKE) download-model-m2m-small; \
				fi; \
			elif [ "$(BABELTRON_MODEL_TYPE)" = "nllb" ]; then \
				if [ "$(MODEL_SIZE)" = "small" ]; then \
					$(MAKE) download-model-nllb-small; \
				elif [ "$(MODEL_SIZE)" = "large" ]; then \
					$(MAKE) download-model-nllb-large; \
				else \
					echo "Invalid model size: $(MODEL_SIZE). Using small model."; \
					$(MAKE) download-model-nllb-small; \
				fi; \
			else \
				echo "Invalid model type: $(BABELTRON_MODEL_TYPE). Using m2m small model."; \
				$(MAKE) download-model-m2m-small; \
			fi; \
		else \
			echo "Aborting."; \
			exit 1; \
		fi; \
	fi
	@echo "Running Docker container..."
	@docker run -p $(PORT):$(PORT) -v $(shell pwd)/$(MODEL_PATH):/models -e MODEL_PATH=/models -e BABELTRON_BABELTRON_MODEL_TYPE=$(BABELTRON_MODEL_TYPE) -e PORT=$(PORT) $(IMAGE_NAME):latest

docker-up: ## Build and start services with docker compose
	@echo "Checking for model files..."
	@if [ ! -d "$(MODEL_PATH)" ] || [ -z "$(shell ls -A $(MODEL_PATH) 2>/dev/null)" ]; then \
		echo "No model files found in $(MODEL_PATH) directory."; \
		read -p "Do you want to download the $(BABELTRON_MODEL_TYPE) $(MODEL_SIZE) model now? (y/n) " answer; \
		if [ "$$answer" = "y" ]; then \
			mkdir -p $(MODEL_PATH); \
			echo "Downloading $(BABELTRON_MODEL_TYPE) $(MODEL_SIZE) model..."; \
			if [ "$(BABELTRON_MODEL_TYPE)" = "m2m" ]; then \
				if [ "$(MODEL_SIZE)" = "small" ]; then \
					$(MAKE) download-model-m2m-small; \
				elif [ "$(MODEL_SIZE)" = "medium" ]; then \
					$(MAKE) download-model-m2m-medium; \
				elif [ "$(MODEL_SIZE)" = "large" ]; then \
					$(MAKE) download-model-m2m-large; \
				else \
					echo "Invalid model size: $(MODEL_SIZE). Using small model."; \
					$(MAKE) download-model-m2m-small; \
				fi; \
			elif [ "$(BABELTRON_MODEL_TYPE)" = "nllb" ]; then \
				if [ "$(MODEL_SIZE)" = "small" ]; then \
					$(MAKE) download-model-nllb-small; \
				elif [ "$(MODEL_SIZE)" = "large" ]; then \
					$(MAKE) download-model-nllb-large; \
				else \
					echo "Invalid model size: $(MODEL_SIZE). Using small model."; \
					$(MAKE) download-model-nllb-small; \
				fi; \
			else \
				echo "Invalid model type: $(BABELTRON_MODEL_TYPE). Using m2m small model."; \
				$(MAKE) download-model-m2m-small; \
			fi; \
		else \
			echo "Model download skipped. Container may not work properly."; \
		fi; \
	fi
	@echo "Building and starting services with docker compose..."
	@BABELTRON_MODEL_TYPE=$(BABELTRON_MODEL_TYPE) docker compose up -d --build
	@echo "Services started successfully. API available at http://localhost:8000"
	@echo "API documentation available at http://localhost:8000/docs"

docker-down:
	@echo "Stopping docker compose services..."
	@docker compose down

docker compose-down: ## Stop Docker Compose services
	@echo "Stopping Docker Compose services..."
	@PORT=$(PORT) docker compose down
	@echo "Services stopped successfully."

pre-commit-install:
	pip install pre-commit
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

docker compose-up: ## Start services with Docker Compose
	@echo "Starting services with Docker Compose..."
	@PORT=$(PORT) docker compose up -d
	@echo "Services started successfully. API is available at http://localhost:$(PORT)/api/docs"
