# Use the official slim Python base image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

# Set workspace directory
WORKDIR /app

# Install system dependencies needed for OpenCV, PyTorch, and metrology
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and uv.lock to install dependencies first (caching layer)
COPY pyproject.toml uv.lock ./

# Install dependencies using uv into system environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r pyproject.toml

# Copy project source files
COPY . .

# Expose ports: 8501 (Dashboard), 9090 (ML Backend)
EXPOSE 8501 9090

# Default command
CMD ["python", "main.py", "dashboard"]
