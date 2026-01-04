#!/bin/bash

# Initialize variables
REBUILD=false
APP_NAME="paper-rag-app"

# Check for arguments
for arg in "$@"
do
    if [ "$arg" == "-Rebuild" ] || [ "$arg" == "--rebuild" ]; then
        REBUILD=true
    fi
done

# --- 1. Detect Hardware ---
if command -v nvidia-smi &> /dev/null; then
    TAG="gpu"
    BUILD_ARG="gpu"
    echo -e "\033[0;32m✅ NVIDIA GPU Detected (Mode: GPU)\033[0m"
else
    TAG="cpu"
    BUILD_ARG="cpu"
    echo -e "\033[0;33m⚠️  No NVIDIA GPU found (Mode: CPU)\033[0m"
fi

IMAGE_FULL_NAME="${APP_NAME}:${TAG}"

# --- 2. Build Strategy ---
IMAGE_EXISTS=$(docker images -q "$IMAGE_FULL_NAME" 2> /dev/null)

if [ "$REBUILD" = true ] || [ -z "$IMAGE_EXISTS" ]; then
    echo -e "\033[0;36m🛠️  Building Image ($IMAGE_FULL_NAME)...\033[0m"
    docker build -t "$IMAGE_FULL_NAME" -f Dockerfile --build-arg DEVICE_TYPE="$BUILD_ARG" .
    if [ $? -ne 0 ]; then
        echo -e "\033[0;31m❌ Build Failed!\033[0m"
        exit 1
    fi
else
    echo -e "\033[0;37m⏩ Image found! Skipping build. (Use -Rebuild to force update)\033[0m"
fi

# --- 3. Determine Paths & OS ---
echo -e "\033[0;32m🚀 Starting Container with Hot Reload...\033[0m"

# Default (Linux/Mac) settings
WORK_DIR=$(pwd)
CONTAINER_PATH_DB="/app/chroma_db"
CONTAINER_PATH_APP="/app"
USE_WINPTY=false

# Windows (Git Bash) detection
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    WORK_DIR=$(pwd -W)
    CONTAINER_PATH_DB="//app/chroma_db"
    CONTAINER_PATH_APP="//app"
    USE_WINPTY=true
    echo -e "\033[0;36m🪟 Windows Git Bash detected: Adjusting paths...\033[0m"
fi

CMD_ARGS=(
    "run" "--rm" "-it"
    "-p" "8000:8000"
    "-v" "${WORK_DIR}:${CONTAINER_PATH_DB}"
    "-v" "${WORK_DIR}:${CONTAINER_PATH_APP}"
)

# --- NEW: Inject .env file if it exists ---
if [ -f ".env" ]; then
    echo -e "\033[0;36m📄 Found .env file, injecting environment variables...\033[0m"
    CMD_ARGS+=("--env-file" ".env")
fi

if [ "$TAG" == "gpu" ]; then
    CMD_ARGS+=("--gpus" "all")
fi

CMD_ARGS+=("$IMAGE_FULL_NAME")
CMD_ARGS+=(
    "uvicorn"
    "backend.main:app"
    "--host" "0.0.0.0"
    "--port" "8000"
    "--reload"
)

# --- 4. Execute ---
if [ "$USE_WINPTY" = true ] && command -v winpty &> /dev/null; then
    MSYS_NO_PATHCONV=1 winpty docker "${CMD_ARGS[@]}"
else
    docker "${CMD_ARGS[@]}"
fi