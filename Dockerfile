FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /usr/local/bin/

WORKDIR /app

# libgomp1 is required by implicit (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Install dependencies (separate layer so it's cached unless lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application source
COPY console.py .
COPY als/ als/

EXPOSE 8000

ENTRYPOINT ["/app/.venv/bin/python", "console.py"]
# Default: serve. Override with "train" for the training container.
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
