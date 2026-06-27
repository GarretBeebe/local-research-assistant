from __future__ import annotations

import asyncio
import secrets
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

import requests

import config

_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".md", ".txt"})
_MAX_FILE_BYTES: int = 10 * 1024 * 1024        # 10 MB before read
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


class JobStatus(TypedDict):
    status: str   # "queued" | "indexing" | "done" | "failed"
    filename: str
    error: str


# Process-local job registry — invisible across uvicorn workers.
# Run with --workers 1 (the default in Dockerfile CMD).
_jobs: dict[str, JobStatus] = {}


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
    if size > _MAX_FILE_BYTES:
        raise IngestError(
            f"File size {size:,} bytes exceeds 10 MB limit"
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
        header = path.read_bytes()[:_MIME_MAGIC_BYTES]
        detected = magic.from_buffer(header, mime=True)
        expected_prefix = _MIME_FOR_EXT[ext]
        if not detected.startswith(expected_prefix):
            raise IngestError(
                f"MIME type mismatch: file declared as {ext!r} but detected as {detected!r}"
            )
    except ImportError:
        print(
            "Warning: python-magic not available; skipping MIME/magic-byte validation",
            file=sys.stderr,
        )


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
        print(
            f"[ingest] TODO: RAG_INGEST_URL not configured "
            f"— skipping vector index for {filename!r}",
            file=sys.stderr,
        )
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
        print(
            f"[ingest] TODO: GRAPH_RAG_INGEST_URL not configured "
            f"— skipping graph index for {filename!r}",
            file=sys.stderr,
        )
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


async def run_background_indexing(job_id: str, path: Path) -> None:
    filename = path.name
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
    finally:
        path.unlink(missing_ok=True)


def create_job(filename: str) -> str:
    job_id = secrets.token_hex(8)
    _jobs[job_id] = {"status": "queued", "filename": filename, "error": ""}
    return job_id


def get_job(job_id: str) -> JobStatus | None:
    return _jobs.get(job_id)
