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

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and assets
COPY assets/ ./assets/
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY index.html .
COPY server.py .
COPY "IndiaWalls RAG Knowledge Base.md" .
COPY ARCHITECTURE.md .

# Download STT models
RUN python scripts/download_models.py

EXPOSE 8765

CMD ["python", "server.py"]
