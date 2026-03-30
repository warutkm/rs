# Phase 12 — FastAPI Dockerfile
# Base: python:3.10-slim  |  EXPOSE 8000
# Matches workflow 12.7

FROM python:3.10-slim

# System deps (gcc needed for implicit / scipy wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Working directory inside container (all paths in main.py use /app/...)
WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-base-v2')"

# Copy project source so HybridRecommender imports work
# (08_hybrid_engine imports 04_ and 05_ via importlib)
COPY src/ ./src/

# Copy API package
COPY api/ ./api/

# Expose the service port
EXPOSE 8000

# Run uvicorn — main.py is in /app/api/
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
