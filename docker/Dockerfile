FROM python:3.11-slim

# System deps (optional but helpful for builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps first (better caching)
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir .

# Copy source
COPY src /app/src

# Default: show CLI help
CMD ["python", "-m", "src.cli.main", "--help"]
