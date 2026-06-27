FROM python:3.12-slim

# Install stockfish + OpenCV system dependencies
# On Debian, apt puts stockfish at /usr/games/stockfish (not in $PATH)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        stockfish \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (separate layer = faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Tell app where Debian puts stockfish
ENV STOCKFISH_PATH=/usr/games/stockfish

# Railway injects $PORT at runtime; default 8000 for local dev
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
