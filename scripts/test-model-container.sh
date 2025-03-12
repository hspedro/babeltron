#!/bin/bash
set -e

# Default values
IMAGE_TAG="small-latest"
CONTAINER_NAME="babeltron-test"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --container-name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Testing Babeltron container with image tag: $IMAGE_TAG"

# Pull the image
echo "Pulling image ghcr.io/hspedro/babeltron:$IMAGE_TAG..."
docker pull ghcr.io/hspedro/babeltron:$IMAGE_TAG

# Run the container
echo "Starting container $CONTAINER_NAME..."
docker run -d --name $CONTAINER_NAME -p 8000:8000 ghcr.io/hspedro/babeltron:$IMAGE_TAG

# Wait for the container to be ready
echo "Waiting for the container to be ready..."
for i in {1..30}; do
  if curl -s http://localhost:8000/healthz | grep -q "ok"; then
    echo "Container is healthy!"
    break
  fi

  if [ $i -eq 30 ]; then
    echo "Container failed to become healthy within timeout"
    docker logs $CONTAINER_NAME
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    exit 1
  fi

  echo "Waiting for container to be ready... ($i/30)"
  sleep 5
done

# Test translation
echo "Testing translation..."
RESPONSE=$(curl -s -X POST "http://localhost:8000/api/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "src_lang": "en",
    "tgt_lang": "es"
  }')

echo "Translation response: $RESPONSE"

if echo $RESPONSE | grep -q "translation"; then
  echo "✅ Translation test passed!"
else
  echo "❌ Translation test failed!"
  docker logs $CONTAINER_NAME
fi

# Clean up
echo "Cleaning up..."
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

echo "Test completed!"
