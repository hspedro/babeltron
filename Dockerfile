# Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

# Install a specific version of Poetry that supports --no-dev
RUN pip install poetry==2.1.1

# Copy project files
COPY pyproject.toml poetry.lock* README.md ./
COPY babeltron ./babeltron

# Configure poetry to not use a virtual environment
RUN poetry config virtualenvs.create false \
    && poetry install --without dev --no-interaction --no-ansi

FROM python:3.10-slim

WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app/babeltron ./babeltron

ENV PYTHONPATH=/app
ENV MODEL_PATH=/models

# Create a non-root user
RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "babeltron.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
