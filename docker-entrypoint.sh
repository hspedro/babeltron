#!/bin/sh
set -e

# Default to 1 worker if not specified
WORKER_COUNT=${WORKER_COUNT:-1}
# Default port to 8000 if not specified
PORT=${PORT:-8000}

# Convert WORKER_COUNT to integer (handles string values from Kubernetes)
# Use regex to validate if WORKER_COUNT is a valid number
if [ -n "$WORKER_COUNT" ] && [ "$WORKER_COUNT" -eq "$WORKER_COUNT" ] 2>/dev/null; then
    WORKER_COUNT_INT=$WORKER_COUNT
else
    echo "Warning: WORKER_COUNT '$WORKER_COUNT' is not a valid number. Defaulting to 1."
    WORKER_COUNT_INT=1
fi

# Validate PORT is a valid number
if [ -n "$PORT" ] && [ "$PORT" -eq "$PORT" ] 2>/dev/null; then
    echo "Using port: $PORT"
else
    echo "Warning: PORT '$PORT' is not a valid number. Defaulting to 8000."
    PORT=8000
fi

export UVICORN_ACCESS_LOG=0
# If WORKER_COUNT is greater than 1, use Gunicorn
if [ "$WORKER_COUNT_INT" -gt 1 ]; then
    echo "Starting with Gunicorn using $WORKER_COUNT_INT workers on port $PORT"
    exec gunicorn babeltron.app.main:app \
        --workers $WORKER_COUNT_INT \
        --worker-class uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:$PORT \
        --access-logfile /dev/null \
        --error-logfile -
else
    echo "Starting with Uvicorn (single worker) on port $PORT"
    exec uvicorn babeltron.app.main:app --host 0.0.0.0 --port $PORT
fi
