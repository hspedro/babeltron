# Contributing

We welcome contributions to Babeltron! This document provides guidelines for setting up your development environment and ensuring code quality.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/hspedro/babeltron.git
cd babeltron
```

2. Install dependencies:
```bash
make install
```

3. Install pre-commit hooks:
```bash
make pre-commit-install
```

## Pre-commit Hooks

Babeltron uses pre-commit hooks to ensure code quality and consistency. These hooks run automatically when you commit changes and check for:

- Outdated Poetry lock files
- Linting issues
- Trailing whitespace and file formatting issues
- YAML syntax errors
- Merge conflicts

To run the pre-commit hooks manually on all files:

```bash
pre-commit run --all-files
```

## Code Style

Babeltron follows these code style guidelines:

- [Black](https://github.com/psf/black) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting

You can run these checks manually with:

```bash
make lint
```

## Testing

Before submitting a pull request, make sure all tests pass:

```bash
make test
```

To run tests with coverage reporting:

```bash
make coverage
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting checks
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Running locally

```bash
# Run with default M2M100 model
make serve

# Set environment variables to use NLLB model
export MODEL_TYPE=nllb
make serve
```

The server has auto-reload enabled, so changes to the code will be reflected immediately.

## Working with Different Model Types

Babeltron supports two types of translation models:

1. **M2M100** - Meta's Multilingual Machine Translation model
2. **NLLB** (No Language Left Behind) - Meta's newer translation model with better support for low-resource languages

When developing features that interact with the translation models, it's important to test with both model types:

```bash
# Download and test with M2M100 model
make download-model-m2m-small
make test

# Download and test with NLLB model
make download-model-nllb-small
export MODEL_TYPE=nllb
make test
```

Note that NLLB models use different language codes than M2M100 models. For example, "en" in M2M100 corresponds to "eng_Latn" in NLLB.
