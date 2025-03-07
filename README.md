# Babel Tron

[![Tests](https://github.com/hspedro/babeltron/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/hspedro/babeltron/actions/workflows/test.yml)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/hspedro/babeltron)](https://github.com/hspedro/babeltron/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.1-green)](https://github.com/hspedro/babeltron/releases)
[![PyPI version](https://badge.fury.io/py/babeltron.svg)](https://badge.fury.io/py/babeltron)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://github.com/hspedro/babeltron/actions/workflows/test.yml)

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

## Downloading Translation Models

Babeltron requires a translation model to function. You can download models of different sizes depending on your needs and hardware constraints:

```bash
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

## Running the API Server

After installing dependencies and downloading a model, you can run the API server locally:

```bash
# Run the server in development mode with auto-reload
make serve

# Or run in production mode (no auto-reload)
make serve-prod
```

The API will be available at http://localhost:8000.

### API Usage Examples

Once the server is running, you can use the translation API:

```bash
# Translate text from English to Spanish
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "src_lang": "en",
    "tgt_lang": "es"
  }'

# Response:
# {"translation":"Hola, ¿cómo estás?"}
```

You can also access the interactive API documentation at http://localhost:8000/docs.

## API Documentation

Babeltron provides interactive API documentation:

- **Swagger UI**: Available at http://localhost:8000/docs when the server is running
- **ReDoc**: Available at http://localhost:8000/redoc for an alternative documentation view

These interactive documentation pages allow you to:
- Explore all available endpoints
- See request and response schemas
- Test the API directly from your browser
- View detailed descriptions of each endpoint and parameter

## Running with Docker

Babeltron can be run as a Docker container, which simplifies deployment and isolates dependencies.

### Building and Running with Docker

```bash
# Start services with Docker Compose
make docker-up
```

The API will be available at http://localhost:8000.

### Stopping Docker Services

```bash
# Stop services
make docker-down
```

### Docker Volume Mounts

The Docker setup mounts the local `./models` directory to `/models` inside the container. This allows you to:

1. Reuse downloaded models between container restarts
2. Use different model sizes without rebuilding the image
3. Persist models even if the container is removed

If no models are found when starting the container, you'll be prompted to download the small model automatically.

## Distributed Tracing

Babeltron supports distributed tracing with OpenTelemetry and Jaeger. The application is configured to send traces to the OpenTelemetry Collector, which forwards them to Jaeger.

### Configuration

Tracing can be configured using the following environment variables:

- `OTLP_MODE`: The OpenTelemetry protocol mode (`otlp-grpc` or `otlp-http`)
- `OTEL_SERVICE_NAME`: The name of the service in traces (default: `babeltron`)
- `OTLP_GRPC_ENDPOINT`: The endpoint for the OpenTelemetry Collector using gRPC (default: `otel-collector:4317`)
- `OTLP_HTTP_ENDPOINT`: The endpoint for the OpenTelemetry Collector using HTTP (default: `http://otel-collector:4318/v1/traces`)

### Accessing Jaeger UI

When running with Docker Compose, you can access the Jaeger UI at:

```
http://localhost:16686
```

### Tracing Features

The distributed tracing implementation provides insights into:

- Request flow through the API
- Detailed timing of translation steps:
  - Tokenization
  - Model inference
  - Decoding
- Error details and context
- Cross-service communication

### Disabling Tracing

To disable tracing, set the `OTLP_GRPC_ENDPOINT` environment variable to `disabled`:

```yaml
environment:
  - OTLP_GRPC_ENDPOINT=disabled
```

## Authentication

Babeltron supports HTTP Basic Authentication to secure API endpoints. When enabled, all API endpoints (except for health checks, metrics, and documentation) require authentication.


### Enabling Authentication

Basic authentication is enabled by setting both environment variables:

```bash
API_USERNAME=your_username
API_PASSWORD=your_password
```

If either variable is not set, authentication will be disabled.

### Using Authentication

When authentication is enabled, clients must include an HTTP Basic Authentication header with each request:

```
Authorization: Basic <base64-encoded-credentials>
```

Where `<base64-encoded-credentials>` is the Base64 encoding of `username:password`.

#### Example with curl

```bash
# Replace 'your_username' and 'your_password' with your credentials
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic $(echo -n 'your_username:your_password' | base64)" \
  -d '{"text": "Hello world", "src_lang": "en", "tgt_lang": "fr"}'
```

or

```bash
# Replace 'your_username' and 'your_password' with your credentials
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -u "your_username:your_password" \
  -d '{
    "text": "Hello world",
    "src_lang": "en",
    "tgt_lang": "fr"
  }'
```

### Excluded Paths

The following paths are excluded from authentication requirements:

- `/docs` - Swagger UI documentation
- `/redoc` - ReDoc documentation
- `/openapi.json` - OpenAPI schema
- `/healthz` - Health check endpoint
- `/readyz` - Readiness check endpoint
- `/metrics` - Prometheus metrics endpoint

## Contributing

Install pre-commit hooks with `make pre-commit-install` and refer to the [CONTRIBUTING.md](docs/CONTRIBUTING.md) file for more information.

## License

MIT License

Copyright (c) 2025 Pedro Soares
