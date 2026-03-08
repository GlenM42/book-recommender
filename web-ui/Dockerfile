FROM python:3.12-slim
WORKDIR /app

# Install uv the same way the root Dockerfile does
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /usr/local/bin/

# Install only what the web-ui needs — no implicit, no numpy, no C compiler required
RUN uv pip install --system \
    "fastapi>=0.135.1" \
    "uvicorn[standard]>=0.41.0" \
    "jinja2>=3" \
    "httpx>=0.28" \
    "python-multipart>=0.0.20"

COPY web-ui/ ./web-ui/

WORKDIR /app/web-ui
EXPOSE 8999
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8999"]
