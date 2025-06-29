# Multi-stage Dockerfile for SmartRAG
# This Dockerfile creates an optimized production image with proper caching and security

ARG PYTHON_VERSION=3.11-slim
ARG APP_USER=smartrag
ARG APP_UID=1000
ARG APP_GID=1000

# ================================
# Base stage: Common dependencies
# ================================
FROM python:${PYTHON_VERSION} as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG APP_USER
ARG APP_UID
ARG APP_GID
RUN groupadd -g ${APP_GID} ${APP_USER} && \
    useradd -u ${APP_UID} -g ${APP_GID} -m -s /bin/bash ${APP_USER}

# ================================
# Builder stage: Install Python dependencies with uv
# ================================
FROM base as builder

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create app directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies with uv
RUN uv sync --frozen --no-dev

# ================================
# Development stage: Full development environment
# ================================
FROM base as development

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    less \
    tree \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Make sure to use venv
ENV PATH="/app/.venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy source code
COPY --chown=${APP_USER}:${APP_USER} . .

# Install package in development mode with uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN uv pip install -e .

# Change to non-root user
USER ${APP_USER}

# Default command for development
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ================================
# Production stage: Optimized runtime
# ================================
FROM base as production

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Make sure to use venv
ENV PATH="/app/.venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy source code with proper ownership
COPY --chown=${APP_USER}:${APP_USER} . .

# Install package with uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN uv pip install .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache && \
    chown -R ${APP_USER}:${APP_USER} /app/logs /app/data /app/cache

# Security: Remove unnecessary packages and clean up
RUN apt-get purge -y --auto-remove build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Change to non-root user
USER ${APP_USER}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command for production
CMD ["python", "-m", "gunicorn", "src.api.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-connections", "1000", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--preload", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info"]

# ================================
# Testing stage: For running tests
# ================================
FROM development as testing

# Install test dependencies with uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN uv sync --frozen

# Set test environment
ENV ENVIRONMENT=test

# Default command for testing
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=src", "--cov-report=term-missing"]

# ================================
# Builder stage for specific ML models
# ================================
FROM base as ml-builder

# Install ML-specific dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libblas3 \
    liblapack3 \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Pre-download ML models (optional optimization)
WORKDIR /app
COPY --chown=${APP_USER}:${APP_USER} scripts/download_models.py ./scripts/
RUN python scripts/download_models.py --cache-dir /app/model_cache

# ================================
# Labels for metadata
# ================================
FROM production as final

LABEL maintainer="SmartRAG Team <team@smartrag.com>" \
      version="1.0.0" \
      description="SmartRAG - Advanced Retrieval-Augmented Generation API" \
      org.opencontainers.image.title="SmartRAG" \
      org.opencontainers.image.description="Advanced RAG system with hybrid search and caching" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="SmartRAG Team" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/smartrag/smartrag" \
      org.opencontainers.image.documentation="https://smartrag.readthedocs.io"