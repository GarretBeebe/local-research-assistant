import ipaddress
import os
import sys
from pathlib import Path

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

GRAPH_RAG_BASE_URL: str = os.getenv("GRAPH_RAG_BASE_URL", "http://localhost:8001")
GRAPH_RAG_API_KEY: str = os.getenv("GRAPH_RAG_API_KEY", "")
GRAPH_RAG_MODEL: str = os.getenv("GRAPH_RAG_MODEL", "qwen2.5:14b")

PLANNER_MODEL: str = "qwen2.5:14b"
RESEARCHER_MODEL: str = "llama3.1:8b"
SYNTHESIZER_MODEL: str = "qwen2.5:7b"
CRITIC_MODEL: str = "qwen2.5:3b"

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


# ── Web server ────────────────────────────────────────────────────────────────

HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = _parse_int_env("PORT", 8080)
API_KEY: str = os.getenv("API_KEY", "")
ALLOW_INSECURE_LOCALONLY: bool = os.getenv("ALLOW_INSECURE_LOCALONLY", "false").lower() == "true"
ADMIN_PASSWORD_HASH: str = os.getenv("ADMIN_PASSWORD_HASH", "")
TRUSTED_PROXY_IPS: list[str] = [
    ip.strip() for ip in os.getenv("TRUSTED_PROXY_IPS", "").split(",") if ip.strip()
]
UPLOAD_STAGING_DIR: Path = Path(os.getenv("UPLOAD_STAGING_DIR", "./staging"))

# TODO: confirm ingestion endpoint signatures with rag-system and local-graph-rag
RAG_INGEST_URL: str = os.getenv("RAG_INGEST_URL", "")
GRAPH_RAG_INGEST_URL: str = os.getenv("GRAPH_RAG_INGEST_URL", "")

def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_server() -> None:
    """Server-specific startup validation. Call from the FastAPI lifespan, not research.py."""
    if ALLOW_INSECURE_LOCALONLY:
        if not _is_loopback(HOST):
            raise SystemExit(
                f"ALLOW_INSECURE_LOCALONLY=true requires HOST to be a loopback address, "
                f"got {HOST!r}"
            )
        cors = os.getenv("CORS_ORIGINS", "")
        if cors.strip() == "*":
            raise SystemExit("ALLOW_INSECURE_LOCALONLY=true is incompatible with CORS_ORIGINS=*")

    if not ALLOW_INSECURE_LOCALONLY and not API_KEY:
        raise SystemExit(
            "Ambiguous security config: set API_KEY (min 32 chars) for bearer-token auth, "
            "or set ALLOW_INSECURE_LOCALONLY=true to restrict to loopback only. "
            "Refusing to start without one of these."
        )

    if API_KEY and len(API_KEY) < 32:
        raise SystemExit(f"API_KEY must be at least 32 characters, got {len(API_KEY)}")

    if not ADMIN_PASSWORD_HASH:
        raise SystemExit(
            "ADMIN_PASSWORD_HASH is required. Generate with:\n"
            "  python -c \"import bcrypt; "
            "print(bcrypt.hashpw(b'yourpass', bcrypt.gensalt()).decode())\""
        )


def validate() -> None:
    """Explicit startup validation. Call from the application entry point, not at import time."""
    configured = {PLANNER_MODEL, RESEARCHER_MODEL, SYNTHESIZER_MODEL, CRITIC_MODEL}
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

    if not GRAPH_RAG_API_KEY:
        print(
            "Warning: GRAPH_RAG_API_KEY not set — assuming ALLOW_INSECURE_LOCALONLY=true",
            file=sys.stderr,
        )
    try:
        requests.get(f"{GRAPH_RAG_BASE_URL}/healthz", timeout=5).raise_for_status()
    except Exception as e:
        raise SystemExit(f"Graph RAG not reachable at {GRAPH_RAG_BASE_URL}: {e}") from e
