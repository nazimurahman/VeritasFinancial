# =============================================================================
# VERITASFINANCIAL - DOCKERFILE
# =============================================================================
# Multi-stage Docker build for fraud detection system
# Optimized for production deployment with GPU support
# =============================================================================

# =============================================================================
# STAGE 1: BUILDER
# =============================================================================
# This stage installs dependencies and prepares the application
FROM python:3.10-slim as builder

# Set build arguments
ARG CUDA_VERSION=11.8
ARG TORCH_VERSION=2.0.1
ARG PYTHON_VERSION=3.10

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /build

# Copy dependency files
COPY requirements.txt .
COPY requirements-dev.txt .
COPY pyproject.toml .
COPY setup.py .
COPY VERSION .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# =============================================================================
# STAGE 2: GPU BUILDER (Optional)
# =============================================================================
# This stage adds GPU support if needed
FROM builder as gpu-builder

# Install CUDA support
RUN pip install --no-cache-dir torch==${TORCH_VERSION}+cu${CUDA_VERSION} \
    --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}

# =============================================================================
# STAGE 3: DEVELOPMENT IMAGE
# =============================================================================
# For development and testing
FROM python:3.10-slim as development

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    ENVIRONMENT=development

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq-dev \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/cache && \
    chown -R appuser:appuser /app/data /app/models /app/logs /app/cache

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.deployment.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# STAGE 4: PRODUCTION IMAGE (CPU)
# =============================================================================
# Optimized for CPU production deployment
FROM python:3.10-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    WORKERS=4

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only necessary application files
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser VERSION .
COPY --chown=appuser:appuser pyproject.toml .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/cache && \
    chown -R appuser:appuser /app/data /app/models /app/logs /app/cache

# Remove development dependencies
RUN pip uninstall -y -r requirements-dev.txt || true

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with gunicorn for production
CMD ["gunicorn", "src.deployment.api.wsgi:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# =============================================================================
# STAGE 5: PRODUCTION IMAGE (GPU)
# =============================================================================
# Optimized for GPU production deployment
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as production-gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    curl \
    libpq-dev \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy virtual environment from GPU builder
COPY --from=gpu-builder --chown=appuser:appuser /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application files
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser VERSION .
COPY --chown=appuser:appuser pyproject.toml .

# Create directories
RUN mkdir -p /app/data /app/models /app/logs /app/cache && \
    chown -R appuser:appuser /app/data /app/models /app/logs /app/cache

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with gunicorn
CMD ["gunicorn", "src.deployment.api.wsgi:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# =============================================================================
# STAGE 6: TRAINING IMAGE
# =============================================================================
# Specialized image for model training
FROM production-gpu as training

# Set environment variables for training
ENV TRAINING_MODE=1 \
    WANDB_API_KEY=${WANDB_API_KEY}

# Install additional training dependencies
RUN pip install --no-cache-dir \
    wandb \
    tensorboard \
    optuna

# Copy training scripts and data
COPY --chown=appuser:appuser notebooks/ ./notebooks/
COPY --chown=appuser:appuser tests/ ./tests/
COPY --chown=appuser:appuser scripts/train_*.py ./scripts/

# Create directory for training artifacts
RUN mkdir -p /app/checkpoints /app/tensorboard

# Default command for training
CMD ["python", "scripts/train_model.py", "--config", "configs/model_config.yaml"]

# =============================================================================
# STAGE 7: API-ONLY IMAGE (Lightweight)
# =============================================================================
# Minimal image for serving only
FROM python:3.10-slim as api

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    ENVIRONMENT=production

# Install minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy only API dependencies
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only API code
COPY --chown=appuser:appuser src/deployment/api/ ./src/deployment/api/
COPY --chown=appuser:appuser src/inference/ ./src/inference/
COPY --chown=appuser:appuser src/utils/ ./src/utils/
COPY --chown=appuser:appuser configs/api_config.yaml ./configs/

# Create models directory
RUN mkdir -p /app/models && chown -R appuser:appuser /app/models

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.deployment.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# DEFAULT STAGE
# =============================================================================
# Use production as default
FROM production

# =============================================================================
# BUILD INSTRUCTIONS
# =============================================================================
# Build commands:
#   # Build CPU production image
#   docker build --target production -t veritasfinancial:latest .
#   
#   # Build GPU production image
#   docker build --target production-gpu -t veritasfinancial:gpu-latest .
#   
#   # Build development image
#   docker build --target development -t veritasfinancial:dev .
#   
#   # Build with specific CUDA version
#   docker build --build-arg CUDA_VERSION=11.8 --target production-gpu -t veritasfinancial:cuda11.8 .
#   
#   # Build lightweight API image
#   docker build --target api -t veritasfinancial:api-latest .
#   
#   # Build training image
#   docker build --target training -t veritasfinancial:training .
#   
#   # Build with build args
#   docker build --build-arg TORCH_VERSION=2.0.1 --target production-gpu -t veritasfinancial:custom .
# =============================================================================

# =============================================================================
# END OF DOCKERFILE
# =============================================================================