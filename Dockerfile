# Use a lightweight Python base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# UPDATE THIS SECTION
# Install system dependencies (added libgl1 and libglib2.0-0 for OpenCV/Docling)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Define Build Argument
ARG DEVICE_TYPE=cpu

# ... (Rest of the file remains exactly the same)
RUN if [ "$DEVICE_TYPE" = "gpu" ]; then \
        echo "Building for NVIDIA GPU (CUDA Support)"; \
        pip install torch torchvision torchaudio; \
    else \
        echo "Building for CPU (Lightweight)"; \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

RUN pip install \
    chromadb \
    tqdm \
    docling \
    numpy \
    transformers \
    fastapi \
    uvicorn \
    python-multipart \
    requests

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]