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
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
