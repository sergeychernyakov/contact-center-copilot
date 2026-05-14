FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for FAISS + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install -e .

# Copy source
COPY src ./src
COPY data ./data
COPY scripts ./scripts

# Generate sample data at build time so the container has something to demo
RUN python scripts/generate_sample_data.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "src/ui/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
