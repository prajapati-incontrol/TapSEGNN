# Minimal PyTorch + CUDA image
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Speed optimizations
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install dependencies
RUN pip install -r requirements.txt

# Install extra dev tools for Jupyter etc.
RUN pip install -r requirements-dev.txt

# Copy project files
COPY src/ src/
COPY config/ config/
COPY main.py .
COPY utils/ utils/

# Create non-root user
RUN useradd -m -u 1000 scientist && chown -R scientist:scientist /app
USER scientist

# Expose Jupyter port
EXPOSE 8888

# Default entrypoint (can be overridden in docker-compose)
CMD ["python", "main.py"]
