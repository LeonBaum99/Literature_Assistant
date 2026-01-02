#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
GRAY='\033[1;30m'
NC='\033[0m' # No Color

APP_NAME="paper-rag-app"
REBUILD=false

# Parse arguments (Check for -Rebuild or --rebuild)
for arg in "$@"
do
    case $arg in
        -Rebuild|--rebuild)
        REBUILD=true
        shift # Remove --rebuild from processing
        ;;
    esac
done

# --- 1. Detect Hardware ---
if command -v nvidia-smi &> /dev/null; then
    TAG="gpu"
    BUILD_ARG="gpu"
    echo -e "${GREEN}NVIDIA GPU Detected (Mode: GPU)${NC}"
else
    TAG="cpu"
    BUILD_ARG="cpu"
    echo -e "${YELLOW}No NVIDIA GPU found (Mode: CPU)${NC}"
fi

IMAGE_FULL_NAME="${APP_NAME}:${TAG}"

# --- 2. Build Strategy ---
# Check if image exists
if [[ "$(docker images -q $IMAGE_FULL_NAME 2> /dev/null)" == "" ]]; then
    IMAGE_EXISTS=false
else
    IMAGE_EXISTS=true
fi

# Determine if we should build
if [ "$IMAGE_EXISTS" = false ] || [ "$REBUILD" = true ]; then
    echo -e "${CYAN}Building Image ($IMAGE_FULL_NAME)...${NC}"

    docker build -t "$IMAGE_FULL_NAME" -f Dockerfile --build-arg DEVICE_TYPE="$BUILD_ARG" .

    # Check if build succeeded
    if [ $? -ne 0 ]; then
        echo -e "${RED}Build Failed!${NC}"
        exit 1
    fi
else
    echo -e "${GRAY}Image found! Skipping build. (Use --rebuild to force update)${NC}"
fi

# --- 3. Run Container (Dev Mode) ---
echo -e "${GREEN}Starting Container with Hot Reload...${NC}"

# Construct Docker Arguments Array
DOCKER_ARGS=(
    run --rm -it
    -p 8000:8000
    # MOUNT 1: Persist the Database
    -v "$(pwd)/chroma_db:/app/chroma_db"
    # MOUNT 2: Sync Source Code (Host -> Container)
    -v "$(pwd):/app"
)

# Add GPU flags if necessary
if [ "$TAG" == "gpu" ]; then
    DOCKER_ARGS+=(--gpus all)
fi

# Add Image Name
DOCKER_ARGS+=("$IMAGE_FULL_NAME")

# OVERRIDE COMMAND: Add --reload so Uvicorn restarts on file save
DOCKER_ARGS+=(uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload)

# Execute Docker
docker "${DOCKER_ARGS[@]}"