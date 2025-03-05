# Babel Tron

[![Tests](https://github.com/hspedro/babeltron/actions/workflows/test.yml/badge.svg)](https://github.com/hspedro/babeltron/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/babeltron.svg)](https://badge.fury.io/py/babeltron)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python-based REST API that leverages single multilingual models like mBERT to
provide efficient text translation services. Babeltron exposes a simple interface
for translating text between multiple languages, making powerful neural machine
translation accessible through straightforward API endpoints.

## Features

- Receives a text, source language and destination language, then returns the text
  translated

## Requirements

- Python 3.9 or higher
- Poetry

## Installation

### Installing Poetry

This project uses Poetry for dependency management. To install Poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Installing Project Dependencies

```bash
make install
```

## Downloading the Translation Model

```
# Download the small model (418M parameters, ~1GB disk space)
make download-model

# Or download medium model (1.2B parameters, ~2.5GB disk space)
make download-model-medium

# Or download large model (12B parameters, ~24GB disk space)
make download-model-large
```

### Model Size Considerations

- **Small (418M)**: ~1GB disk space, less memory required, faster but less accurate
- **Medium (1.2B)**: ~2.5GB disk space, moderate memory requirements
- **Large (12B)**: ~24GB disk space, requires significant RAM/GPU memory

Choose based on your hardware constraints and translation quality requirements.

## Model Workflow

The complete workflow for using Babeltron involves:

1. **Install dependencies**: `make install`
2. **Download a model**: `make download-model` (or medium/large variants)
3. **Package the model**: `make package-model`
4. **Serve the model**: `make serve-model`

Each step builds on the previous one. You must download a model before packaging it, and package it before serving it.

## Development Commands

The project includes several helpful make commands:

- `make install` - Install project dependencies
- `make test` - Run tests
- `make lint` - Run linters (flake8, isort, black)
- `make format` - Format code with isort and black
- `make coverage` - Run tests with coverage report
- `make coverage-html` - Generate HTML coverage report

## Testing and Code Coverage

### Running Tests with Coverage

To run tests with coverage reporting:

```bash
make coverage
```

For a detailed HTML coverage report:

```bash
make coverage-html
```

The HTML report will be generated in the `htmlcov` directory.

### Coverage Configuration

The project uses a `.coveragerc` file to configure coverage settings. This ensures consistent coverage reporting across different environments.

## License

MIT License

Copyright (c) 2025 Pedro Soares
