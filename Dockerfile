# Base image with Python
FROM python:3.10-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/

# Copy Poetry configuration files
COPY README.md pyproject.toml poetry.lock* ./

# Copy application code
COPY babeltron/ ./babeltron/

# Configure Poetry to not use virtualenvs in Docker
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --without dev --no-interaction --no-ansi

# Set environment variables
ENV MODEL_PATH=/models
ENV PYTHONPATH=/app
ENV WORKERS=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV MODEL_COMPRESSION_ENABLED=true

# Expose the port the app runs on
EXPOSE 8000

# Create a script to start the application with the specified number of workers
RUN echo '#!/bin/bash\n\
echo "Starting with $WORKERS workers"\n\
echo "Model compression: $MODEL_COMPRESSION_ENABLED"\n\
if [ "$WORKERS" -eq "1" ]; then\n\
  # Single worker mode uses uvicorn directly\n\
  exec uvicorn babeltron.app.main:app --host $HOST --port $PORT\n\
else\n\
  # Multi-worker mode uses gunicorn with uvicorn workers\n\
  exec gunicorn babeltron.app.main:app \\\n\
    --workers $WORKERS \\\n\
    --worker-class uvicorn.workers.UvicornWorker \\\n\
    --bind $HOST:$PORT\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Command to run the application
CMD ["/app/start.sh"]
