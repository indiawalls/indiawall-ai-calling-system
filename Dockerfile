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

# Upgrade pip and packaging tools for modern wheel resolution
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install lightweight PyTorch CPU (--extra-index-url preserves PyPI for flit_core and dependencies)
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
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
