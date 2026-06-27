FROM python:3.12-slim

# libmagic is required by python-magic
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN adduser --system --no-create-home --group appuser

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --no-dev

COPY . .

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

# Single worker required — job status is tracked in process memory.
# Shell form so uvicorn's --host follows the HOST env var (must match config.HOST for
# validate_server() to agree on whether the binding is loopback-only).
CMD ["sh", "-c", "uv run uvicorn api:app --host \"${HOST:-127.0.0.1}\" --port \"${PORT:-8080}\" --workers 1"]
