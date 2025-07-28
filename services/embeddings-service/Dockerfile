FROM python:3.11-slim

# Install system dependencies for gRPC and compilation
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install UV for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy service files
COPY pyproject.toml .
COPY main.py .
COPY triton_client.py .
COPY cache.py .

# Install dependencies (no shared utilities needed)
RUN uv pip install --system --no-cache -e .

EXPOSE 8001

# Health check using the service's own health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/embeddings/health', timeout=5).raise_for_status()"

# Run the FastAPI service
CMD ["python", "main.py"]