import os

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Phase 2: dedicated Ollama instances per role. Both default to OLLAMA_URL so Phase 1
# single-instance setups work without any .env changes.
OLLAMA_PLANNER_URL: str = os.getenv("OLLAMA_PLANNER_URL", OLLAMA_URL)
OLLAMA_RESEARCHER_URL: str = os.getenv("OLLAMA_RESEARCHER_URL", OLLAMA_URL)

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


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"Invalid value for {name}={raw!r} — must be an integer") from None


# Resource governor
MAX_CONCURRENT_RESEARCHERS: int = _parse_int_env("MAX_CONCURRENT_RESEARCHERS", 2)
MEMORY_PRESSURE_THRESHOLD_MB: int = _parse_int_env("MEMORY_PRESSURE_THRESHOLD_MB", 6144)


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

    for url in dict.fromkeys([OLLAMA_URL, OLLAMA_PLANNER_URL, OLLAMA_RESEARCHER_URL]):
        try:
            requests.get(f"{url}/api/tags", timeout=5).raise_for_status()
        except Exception as e:
            raise SystemExit(f"Ollama not reachable at {url}: {e}") from e
