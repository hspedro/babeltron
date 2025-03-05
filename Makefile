.PHONY: check-poetry install test lint format help system-deps coverage coverage-html package-model serve-model download-model download-model-small download-model-medium download-model-large clean

# Extract target descriptions from comments
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

POETRY := $(shell command -v poetry 2> /dev/null)
UNAME := $(shell uname -s)

# Model packaging variables
MODEL_NAME = m2m_translation
MODEL_VERSION = 1.0
MODEL_DIR = babeltron/model
HANDLER_PATH = babeltron/handler/m2m_handler.py
MODEL_STORE = model_store

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

package-model: check-poetry ## Package the M2M translation model for TorchServe
	@echo "Packaging model..."
	@mkdir -p $(MODEL_STORE)
	@echo "Checking model directory contents:"
	@ls -la $(MODEL_DIR)

	@if [ ! -f "$(MODEL_DIR)/pytorch_model.bin" ] && [ ! -f "$(MODEL_DIR)/model.safetensors" ]; then \
		echo "Error: Model files not found. Please run 'make download-model' first."; \
		exit 1; \
	fi

	@echo "Using model files from $(MODEL_DIR)"
	@MODEL_FILE=$$(if [ -f "$(MODEL_DIR)/pytorch_model.bin" ]; then echo "$(MODEL_DIR)/pytorch_model.bin"; else echo "$(MODEL_DIR)/model.safetensors"; fi) && \
	echo "Using model file: $$MODEL_FILE" && \
	echo "Checking for required tokenizer files..." && \
	if [ ! -f "$(MODEL_DIR)/sentencepiece.bpe.model" ]; then \
		echo "Warning: sentencepiece.bpe.model not found"; \
	fi && \
	if [ ! -f "$(MODEL_DIR)/tokenizer_config.json" ]; then \
		echo "Warning: tokenizer_config.json not found"; \
	fi && \
	EXTRA_FILES="$(MODEL_DIR)/config.json" && \
	if [ -f "$(MODEL_DIR)/tokenizer_config.json" ]; then \
		EXTRA_FILES="$$EXTRA_FILES,$(MODEL_DIR)/tokenizer_config.json"; \
	fi && \
	if [ -f "$(MODEL_DIR)/sentencepiece.bpe.model" ]; then \
		EXTRA_FILES="$$EXTRA_FILES,$(MODEL_DIR)/sentencepiece.bpe.model"; \
	fi && \
	if [ -f "$(MODEL_DIR)/tokenizer.json" ]; then \
		EXTRA_FILES="$$EXTRA_FILES,$(MODEL_DIR)/tokenizer.json"; \
	fi && \
	echo "Using extra files: $$EXTRA_FILES" && \
	echo "Running torch-model-archiver..." && \
	poetry run torch-model-archiver \
		--model-name $(MODEL_NAME) \
		--version $(MODEL_VERSION) \
		--serialized-file $$MODEL_FILE \
		--handler $(HANDLER_PATH) \
		--extra-files "$$EXTRA_FILES" \
		--export-path $(MODEL_STORE) || { echo "Error: torch-model-archiver failed"; exit 1; }
	@echo "Model packaged successfully to $(MODEL_STORE)/$(MODEL_NAME).mar"

serve-model: package-model ## Start TorchServe with the packaged model
	@echo "Starting TorchServe with the model..."
	@poetry run torchserve --start --model-store $(MODEL_STORE) --models $(MODEL_NAME)=$(MODEL_NAME).mar

# Default model size is small (418M)
download-model: download-model-small

# Download specific model sizes
download-model-small:
	@poetry run python scripts/download_models.py --size 418M

download-model-medium:
	@poetry run python scripts/download_models.py --size 1.2B

download-model-large:
	@poetry run python scripts/download_models.py --size 12B

# Clean up
clean:
	rm -rf babeltron/model/m2m100_*
	rm -f babeltron/model/model_config.txt
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
