import os

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")

RAG_BASE_URL: str = os.getenv("RAG_BASE_URL", "")
RAG_INTERNAL_TOKEN: str = os.getenv("RAG_INTERNAL_TOKEN", "")

PLANNER_MODEL: str = "qwen2.5:14b"
RESEARCHER_MODEL: str = "llama3.1:8b"
SYNTHESIZER_MODEL: str = "qwen2.5:7b"

# Chat-capable models only — do not include embedding-only models here.
# The planner chooses from this set; the researcher executes with it.
CHAT_MODELS: frozenset[str] = frozenset({
    "qwen2.5:14b",
    "qwen2.5:7b",
    "llama3.1:8b",
    "qwen2.5:3b",
})

EMBED_MODELS: frozenset[str] = frozenset({
    "nomic-embed-text",
})

ALLOWED_MODELS: frozenset[str] = CHAT_MODELS | EMBED_MODELS

DEBUG_LOG_FULL_PAYLOADS: bool = os.getenv("DEBUG_LOG_FULL_PAYLOADS", "false").lower() == "true"

MAX_QUERY_LENGTH: int = 12_000


def validate() -> None:
    """Explicit startup validation. Call from the application entry point, not at import time."""
    configured = {PLANNER_MODEL, RESEARCHER_MODEL, SYNTHESIZER_MODEL}
    unknown = configured - CHAT_MODELS
    if unknown:
        raise SystemExit(f"Configured models not in CHAT_MODELS: {unknown}")

    if not RAG_BASE_URL:
        raise SystemExit("RAG_BASE_URL is required — set it in .env")

    if not RAG_INTERNAL_TOKEN:
        raise SystemExit("RAG_INTERNAL_TOKEN is required — set it in .env")

    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        raise SystemExit(f"Ollama not reachable at {OLLAMA_URL}: {e}") from e
