FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install UV for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy service files
COPY pyproject.toml .
COPY main.py .

# Install dependencies
RUN uv pip install --system --no-cache -e .

# Install shared utilities from parent directory
COPY ../shared /shared
RUN uv pip install --system --no-cache -e /shared

EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')"

CMD ["python", "main.py"]