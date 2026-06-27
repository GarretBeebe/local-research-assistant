from __future__ import annotations

import asyncio
import hmac
import ipaddress
import secrets
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import bcrypt
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

import config
import db
import ingest

_UI_DIR = Path(__file__).parent / "ui"
_MAX_BUCKET_SIZE = 10_000
_TEMPLATES = Jinja2Templates(directory=str(_UI_DIR))
_SESSION_COOKIE = "session"
_CSRF_HEADER = "X-CSRF-Token"

# Holds references to background tasks so they are not GC'd before completion.
_background_tasks: set[asyncio.Task[Any]] = set()


def _is_valid_ip(ip: str) -> bool:
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


# ── Middleware ─────────────────────────────────────────────────────────────────

class TrustedProxyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, trusted_ips: list[str]) -> None:
        super().__init__(app)
        self._trusted: frozenset[str] = frozenset(trusted_ips)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        client_ip = request.client.host if request.client else ""
        if client_ip in self._trusted:
            forwarded = request.headers.get("X-Forwarded-For", "").split(",")
            real_ip = forwarded[0].strip()
            # Reject non-IP values (empty, hostnames, injected garbage) to
            # prevent rate-limit bypass via a spoofed XFF header.
            if real_ip and _is_valid_ip(real_ip):
                request.state.real_ip = real_ip
                request.state.https = (
                    request.headers.get("X-Forwarded-Proto", "").lower() == "https"
                )
                return await call_next(request)
        # Falls through when XFF is absent/invalid — all such clients share
        # the proxy's rate bucket. Ensure the proxy always forwards XFF.
        request.state.real_ip = client_ip
        request.state.https = False
        return await call_next(request)


class _Bucket:
    __slots__ = ("tokens", "last")

    def __init__(self, capacity: float) -> None:
        self.tokens = capacity
        self.last = time.monotonic()


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: Any,
        general_limit: int = 30,
        general_window: int = 60,
        login_limit: int = 10,
        login_window: int = 60,
    ) -> None:
        super().__init__(app)
        self._gen_rate = general_limit / general_window
        self._gen_cap = float(general_limit)
        self._login_rate = login_limit / login_window
        self._login_cap = float(login_limit)
        self._buckets: dict[str, _Bucket] = {}
        self._login_buckets: dict[str, _Bucket] = {}

    def _consume(self, bucket: _Bucket, rate: float, cap: float) -> bool:
        now = time.monotonic()
        bucket.tokens = min(cap, bucket.tokens + (now - bucket.last) * rate)
        bucket.last = now
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True
        return False

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        ip = getattr(request.state, "real_ip", request.client.host if request.client else "")
        is_login = request.method == "POST" and request.url.path == "/login"
        buckets = self._login_buckets if is_login else self._buckets
        rate = self._login_rate if is_login else self._gen_rate
        cap = self._login_cap if is_login else self._gen_cap
        if ip not in buckets:
            if len(buckets) >= _MAX_BUCKET_SIZE:
                buckets.pop(next(iter(buckets)))  # evict oldest (insertion order)
            buckets[ip] = _Bucket(cap)
        if not self._consume(buckets[ip], rate, cap):
            return Response("Too Many Requests", status_code=429)
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Response:
        response = await call_next(request)
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'self'",
        })
        return response


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):  # type: ignore[misc]
    config.validate_server()
    await asyncio.to_thread(config.validate)
    db.init_db()
    config.UPLOAD_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    yield
    db.prune_expired_sessions()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

# Middleware: last added runs first (outermost). Order: TrustedProxy → RateLimit → SecurityHeaders
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    general_limit=30, general_window=60,
    login_limit=10, login_window=60,
)
app.add_middleware(TrustedProxyMiddleware, trusted_ips=config.TRUSTED_PROXY_IPS)
app.mount("/static", StaticFiles(directory=str(_UI_DIR / "static")), name="static")


# ── Auth helpers ───────────────────────────────────────────────────────────────

def _resolve_session(request: Request) -> dict | None:
    token = request.cookies.get(_SESSION_COOKIE)
    if not token:
        return None
    row = db.get_session(token)
    if row is None:
        return None
    if datetime.now(tz=UTC) >= datetime.fromisoformat(row["expires_at"]):
        db.delete_session(token)
        return None
    return row


def _require_session_api(request: Request) -> dict:
    """For AJAX /ui/* routes — returns 401, not a redirect."""
    row = _resolve_session(request)
    if row is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return row


def _require_bearer(request: Request) -> None:
    # In insecure local-only mode without an API key, /v1/* is open (loopback only).
    # validate_server() guarantees API_KEY is set if we fall through this check.
    if config.ALLOW_INSECURE_LOCALONLY and not config.API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    provided = auth[len("Bearer "):].strip()
    if not hmac.compare_digest(provided.encode(), config.API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def _check_csrf(request: Request, session: dict) -> None:
    client_tok = request.headers.get(_CSRF_HEADER, "")
    if not hmac.compare_digest(client_tok.encode(), session["csrf_token"].encode()):
        raise HTTPException(status_code=403, detail="CSRF token invalid or missing")


def _cookie_secure(request: Request) -> bool:
    return getattr(request.state, "https", False) or request.url.scheme == "https"


# ── Pydantic models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=config.MAX_QUERY_LENGTH)


# ── Shared helpers ─────────────────────────────────────────────────────────────

async def _run_and_store(query: str) -> dict:
    from pipeline import run_pipeline
    result = await run_pipeline(query)
    history_id = db.save_query(
        query=query,
        answer=result.answer,
        confidence=result.confidence,
        critic_passed=result.critic_passed,
        re_planned=result.re_planned,
    )
    return {
        "answer": result.answer,
        "history_id": history_id,
        "confidence": result.confidence,
        "partial": result.partial,
    }


async def _accept_upload(upload: UploadFile) -> str:
    """Save upload to staging, validate, queue background indexing. Returns job_id."""
    raw_name = upload.filename or "upload"
    filename = Path(raw_name).name or "upload"  # strip any path components
    staging = config.UPLOAD_STAGING_DIR
    # Unique prefix prevents concurrent uploads of the same filename from overwriting each other.
    dest = staging / f"{secrets.token_hex(8)}_{filename}"

    contents = await upload.read(ingest.MAX_FILE_BYTES + 1)
    if len(contents) > ingest.MAX_FILE_BYTES:
        _mb = ingest.MAX_FILE_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File exceeds {_mb} MB limit")

    try:
        await asyncio.to_thread(dest.write_bytes, contents)
        await asyncio.to_thread(ingest.validate_upload, dest, staging)
    except ingest.IngestError as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception:
        dest.unlink(missing_ok=True)
        raise

    try:
        job_id = ingest.create_job(filename)
    except ingest.IngestError as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(e)) from e
    task = asyncio.create_task(ingest.run_background_indexing(job_id, dest, filename))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return job_id


# ── Browser UI routes ──────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def get_login(request: Request) -> Response:
    return _TEMPLATES.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def post_login(request: Request) -> Response:
    form = await request.form()
    password = str(form.get("password", ""))
    try:
        valid = await asyncio.to_thread(
            bcrypt.checkpw, password.encode(), config.ADMIN_PASSWORD_HASH.encode()
        )
    except Exception:
        valid = False

    if not valid:
        return _TEMPLATES.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid password"},
            status_code=401,
        )

    token, _csrf = db.create_session()  # CSRF token is embedded in GET / HTML, not this redirect
    resp = RedirectResponse(url="/", status_code=303)
    resp.set_cookie(
        _SESSION_COOKIE,
        token,
        max_age=db._SESSION_TTL_HOURS * 3600,
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(request),
    )
    return resp


@app.post("/logout")
async def post_logout(request: Request) -> Response:
    session = _resolve_session(request)
    if session:
        _check_csrf(request, session)
        db.delete_session(session["token"])
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie(_SESSION_COOKIE, secure=_cookie_secure(request))
    return resp


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request) -> Response:
    session = _resolve_session(request)
    if session is None:
        return RedirectResponse(url="/login", status_code=303)
    return _TEMPLATES.TemplateResponse(
        "index.html",
        {
            "request": request,
            "csrf_token": session["csrf_token"],
            "max_query_length": config.MAX_QUERY_LENGTH,
        },
    )


@app.post("/ui/query")
async def ui_query(request: Request, body: QueryRequest) -> JSONResponse:
    session = _require_session_api(request)
    _check_csrf(request, session)
    return JSONResponse(await _run_and_store(body.query))


@app.post("/ui/ingest")
async def ui_ingest(request: Request, file: UploadFile = File(...)) -> JSONResponse:  # noqa: B008
    session = _require_session_api(request)
    _check_csrf(request, session)
    job_id = await _accept_upload(file)
    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/ui/history")
async def ui_history(request: Request) -> JSONResponse:
    _require_session_api(request)
    return JSONResponse(db.get_history())


@app.delete("/ui/history/{query_id}")
async def ui_delete_history(request: Request, query_id: int) -> JSONResponse:
    session = _require_session_api(request)
    _check_csrf(request, session)
    if not db.delete_query(query_id):
        raise HTTPException(status_code=404, detail="Query not found")
    return JSONResponse({"deleted": query_id})


@app.get("/ui/ingest/status/{job_id}")
async def ui_ingest_status(request: Request, job_id: str) -> JSONResponse:
    _require_session_api(request)
    job = ingest.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job)


# ── API routes (bearer token) ──────────────────────────────────────────────────

@app.post("/v1/query")
async def v1_query(request: Request, body: QueryRequest) -> JSONResponse:
    _require_bearer(request)
    return JSONResponse(await _run_and_store(body.query))


@app.post("/v1/ingest")
async def v1_ingest(request: Request, file: UploadFile = File(...)) -> JSONResponse:  # noqa: B008
    _require_bearer(request)
    job_id = await _accept_upload(file)
    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/v1/history")
async def v1_history(request: Request) -> JSONResponse:
    _require_bearer(request)
    return JSONResponse(db.get_history())


@app.delete("/v1/history/{query_id}")
async def v1_delete_history(request: Request, query_id: int) -> JSONResponse:
    _require_bearer(request)
    if not db.delete_query(query_id):
        raise HTTPException(status_code=404, detail="Query not found")
    return JSONResponse({"deleted": query_id})


@app.get("/v1/ingest/status/{job_id}")
async def v1_ingest_status(request: Request, job_id: str) -> JSONResponse:
    _require_bearer(request)
    job = ingest.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job)
