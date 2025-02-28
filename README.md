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
