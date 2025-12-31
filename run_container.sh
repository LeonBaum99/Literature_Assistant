#!/bin/bash

APP_NAME="paper-rag-app"

# Check if nvidia-smi exists and if a GPU is actually found
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU Detected!"
    echo "🛠️  Building GPU-enabled Container..."

    # Build with DEVICE_TYPE=gpu
    docker build -t ${APP_NAME}:gpu --build-arg DEVICE_TYPE=gpu .

    echo "🚀 Running with GPU support..."
    docker run --rm -it \
        --gpus all \
        -p 8000:8000 \
        -v $(pwd)/chroma_db:/app/chroma_db \
        ${APP_NAME}:gpu

else
    echo "⚠️  No NVIDIA GPU found (or nvidia-smi missing)."
    echo "🛠️  Building CPU-only Container (Lightweight)..."

    # Build with DEVICE_TYPE=cpu
    docker build -t ${APP_NAME}:cpu --build-arg DEVICE_TYPE=cpu .

    echo "🚀 Running in CPU mode..."
    docker run --rm -it \
        -p 8000:8000 \
        -v $(pwd)/chroma_db:/app/chroma_db \
        ${APP_NAME}:cpu
fi