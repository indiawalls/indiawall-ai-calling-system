FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 1. Install lightweight PyTorch CPU first (saves ~8GB of NVIDIA CUDA bloat)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy application code and assets
COPY assets/ ./assets/
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY index.html .
COPY server.py .
COPY ["IndiaWalls RAG Knowledge Base.md", "."]
COPY ARCHITECTURE.md .

# 4. Download local STT models
RUN python scripts/download_models.py

EXPOSE 8765

CMD ["python", "server.py"]
