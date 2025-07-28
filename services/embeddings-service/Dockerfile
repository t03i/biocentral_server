# Use the official uv image with Python 3.11
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Install system dependencies for gRPC
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files first (for better caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies (not the project itself)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# Copy the project source code
COPY . .

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Expose the service port
EXPOSE 8001

# Health check using the service's own health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD uv run python -c "import httpx; httpx.get('http://localhost:8001/embeddings/health', timeout=5).raise_for_status()"

# Run the application using uv
CMD ["uv", "run", "fastapi", "run", "src.embedding_service.main:app", "--host", "0.0.0.0", "--port", "8001"]