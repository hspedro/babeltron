#!/bin/bash
set -e

# Default to 1 worker if not specified
WORKER_COUNT=${WORKER_COUNT:-1}

# If WORKER_COUNT is greater than 1, use Gunicorn
if [ "$WORKER_COUNT" -gt 1 ]; then
    echo "Starting with Gunicorn using $WORKER_COUNT workers"
    # Set environment variable to disable Uvicorn access logs inside Gunicorn workers
    export UVICORN_ACCESS_LOG=0
    exec gunicorn babeltron.app.main:app \
        --workers $WORKER_COUNT \
        --worker-class uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:8000 \
        --access-logfile /dev/null \
        --error-logfile -
else
    echo "Starting with Uvicorn (single worker)"
    exec UVICORN_ACCESS_LOG=0 uvicorn babeltron.app.main:app --host 0.0.0.0 --port 8000
fi
