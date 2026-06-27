from __future__ import annotations

import asyncio
import logging
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict

import requests

import config

_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".md", ".txt"})
MAX_FILE_BYTES: int = 10 * 1024 * 1024         # 10 MB before read
_MAX_EXTRACTED_BYTES: int = 50 * 1024 * 1024   # 50 MB extracted text
_PDF_PAGE_LIMIT: int = 500
_PDF_TIMEOUT_SEC: int = 30
_MIME_MAGIC_BYTES: int = 512

# Expected MIME prefixes per extension (from python-magic)
_MIME_FOR_EXT: dict[str, str] = {
    ".pdf": "application/pdf",
    ".md":  "text/",
    ".txt": "text/",
}

_MAX_QUEUED_JOBS = 20       # max "queued" + "indexing" jobs allowed at once
_JOB_TTL_SECONDS = 3600    # prune completed/failed jobs after 1 hour
_MAX_CONCURRENT_INDEXING = 4
_indexing_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_INDEXING)

_logger = logging.getLogger(__name__)


class JobStatus(TypedDict):
    status: str   # "queued" | "indexing" | "done" | "failed"
    filename: str
    error: str


# Process-local job registry — invisible across uvicorn workers.
# Run with --workers 1 (the default in Dockerfile CMD).
_jobs: dict[str, JobStatus] = {}
_job_created_at: dict[str, float] = {}  # creation timestamps for TTL pruning


class IngestError(ValueError):
    pass


def validate_upload(path: Path, staging_dir: Path) -> None:
    """Validate an uploaded file before accepting it for indexing. Raises IngestError on failure."""
    if path.is_symlink():
        raise IngestError("Symlinks are not permitted")

    try:
        resolved = path.resolve()
        staging_resolved = staging_dir.resolve()
        resolved.relative_to(staging_resolved)
    except ValueError as e:
        raise IngestError("File path escapes the staging directory") from e

    if not path.exists():
        raise IngestError("File not found")

    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise IngestError(
            f"File size {size:,} bytes exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB limit"
        )

    ext = path.suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise IngestError(
            f"Extension {ext!r} not allowed; permitted: {sorted(_ALLOWED_EXTENSIONS)}"
        )

    _check_mime(path, ext)


def _check_mime(path: Path, ext: str) -> None:
    try:
        import magic
    except ImportError:
        _logger.warning("python-magic not available; skipping MIME/magic-byte validation")
        return
    try:
        header = path.read_bytes()[:_MIME_MAGIC_BYTES]
        detected = magic.from_buffer(header, mime=True)
        expected_prefix = _MIME_FOR_EXT[ext]
        if not detected.startswith(expected_prefix):
            raise IngestError(
                f"MIME type mismatch: file declared as {ext!r} but detected as {detected!r}"
            )
    except magic.MagicException as e:
        raise IngestError(f"MIME detection failed: {e}") from e


def _extract_pdf(path: Path) -> str:
    script = (
        "import sys, pypdf; "
        f"reader = pypdf.PdfReader({str(path)!r}); "
        f"pages = reader.pages[:{_PDF_PAGE_LIMIT}]; "
        "text = '\\n'.join(p.extract_text() or '' for p in pages); "
        "sys.stdout.write(text)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=_PDF_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as e:
        raise IngestError(f"PDF extraction timed out after {_PDF_TIMEOUT_SEC}s") from e

    if result.returncode != 0:
        raise IngestError(f"PDF extraction failed: {result.stderr[:500]}")

    text = result.stdout
    if len(text.encode()) > _MAX_EXTRACTED_BYTES:
        raise IngestError("Extracted text exceeds 50 MB limit")

    return text


def _extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _index_vector(filename: str, text: str) -> None:
    # TODO: wire to rag-system ingest endpoint once RAG_INGEST_URL is confirmed
    if not config.RAG_INGEST_URL:
        _logger.warning("RAG_INGEST_URL not configured — skipping vector index for %r", filename)
        return
    try:
        resp = requests.post(
            config.RAG_INGEST_URL,
            json={"filename": filename, "text": text},
            headers={"Authorization": f"Bearer {config.RAG_INTERNAL_TOKEN}"},
            timeout=60,
        )
        resp.raise_for_status()
    except Exception as e:
        raise IngestError(f"Vector RAG ingest failed: {e}") from e


def _index_graph(filename: str, text: str) -> None:
    # TODO: wire to local-graph-rag ingest endpoint once GRAPH_RAG_INGEST_URL is confirmed
    if not config.GRAPH_RAG_INGEST_URL:
        _logger.warning("GRAPH_RAG_INGEST_URL not configured — skipping graph index for %r",
                        filename)
        return
    headers: dict[str, str] = {}
    if config.GRAPH_RAG_API_KEY:
        headers["Authorization"] = f"Bearer {config.GRAPH_RAG_API_KEY}"
    try:
        resp = requests.post(
            config.GRAPH_RAG_INGEST_URL,
            json={"filename": filename, "text": text},
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
    except Exception as e:
        raise IngestError(f"Graph RAG ingest failed: {e}") from e


async def run_background_indexing(job_id: str, path: Path, filename: str) -> None:
    # Semaphore caps concurrent indexing; job waits in "queued" state until a slot opens.
    try:
        async with _indexing_semaphore:
            _jobs[job_id]["status"] = "indexing"
            try:
                text = await asyncio.to_thread(_extract_text, path)
                await asyncio.gather(
                    asyncio.to_thread(_index_vector, filename, text),
                    asyncio.to_thread(_index_graph, filename, text),
                )
                _jobs[job_id]["status"] = "done"
            except IngestError as e:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(e)
            except Exception as e:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = f"Unexpected error: {e}"
    except BaseException:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = "Cancelled"
        raise
    finally:
        path.unlink(missing_ok=True)


def _prune_jobs() -> None:
    now = time.time()
    stale = [
        jid for jid, ts in _job_created_at.items()
        if _jobs.get(jid, {}).get("status") in ("done", "failed") and now - ts > _JOB_TTL_SECONDS
    ]
    for jid in stale:
        _jobs.pop(jid, None)
        _job_created_at.pop(jid, None)


def create_job(filename: str) -> str:
    _prune_jobs()
    active = sum(1 for j in _jobs.values() if j["status"] in ("queued", "indexing"))
    if active >= _MAX_QUEUED_JOBS:
        raise IngestError("Indexing queue is full; try again later")
    job_id = secrets.token_hex(8)
    _jobs[job_id] = {"status": "queued", "filename": filename, "error": ""}
    _job_created_at[job_id] = time.time()
    return job_id


def get_job(job_id: str) -> JobStatus | None:
    return _jobs.get(job_id)
