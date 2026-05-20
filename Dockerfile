# Use the official slim Python base image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder

# Set workspace directory
WORKDIR /app

# Enable bytecode compilation and optimization
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy pyproject.toml and uv.lock to install dependencies first (caching layer)
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (using --system to install into container python)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r pyproject.toml

# Copy project source files
COPY . .

# Final production stage
FROM python:3.10-slim-bookworm

WORKDIR /app

# Copy python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Install system dependencies needed for OpenCV, PyTorch, and metrology
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Expose ports: 8501 (Dashboard), 9090 (ML Backend)
EXPOSE 8501 9090

# Default command
CMD ["python", "main.py", "dashboard"]
