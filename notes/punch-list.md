# Code Review Punch List

Found by `/code-review` on 2026-06-26. All 15 items fixed 2026-06-26.

---

## Critical (security / won't start)

- [x] **Dockerfile + config.py** — `CMD --host 0.0.0.0` disconnected from `config.HOST`.
  **Fixed:** Dockerfile CMD is now shell-form `sh -c '... --host "${HOST:-0.0.0.0}"'` so
  uvicorn's bind address follows the `HOST` env var and agrees with `validate_server()`.

- [x] **docker-compose.yml:10** — `HOST: "0.0.0.0"` caused `validate_server()` SystemExit.
  **Fixed:** Removed `HOST` override from compose `environment` block. `HOST` now comes from
  `.env` only; Dockerfile CMD defaults to `0.0.0.0` when `HOST` is unset.

---

## High (wrong behaviour / auth)

- [x] **api.py ~line 294** — `delete_cookie()` missing `secure=True` on HTTPS.
  **Fixed:** `resp.delete_cookie(_SESSION_COOKIE, secure=_cookie_secure(request))`

- [x] **api.py ~line 184** — bearer token not stripped; trailing space → false 401.
  **Fixed:** `provided = auth[len("Bearer "):].strip()`

- [x] **api.py ~line 234** — `validate_upload` + `write_bytes` blocked event loop; write
  failure orphaned staging file.
  **Fixed:** Both calls wrapped in `asyncio.to_thread` inside a unified `try/except` that
  calls `dest.unlink()` on any failure path.

- [x] **ingest.py:172** — `filename = path.name` captured hex prefix sent to RAG services.
  **Fixed:** `run_background_indexing` now accepts `filename: str` as a parameter; call site
  passes the original filename.

- [x] **ingest.py:84** — `except ImportError` missed `magic.MagicException`.
  **Fixed:** Separated into two `try` blocks: one for `ImportError`, one catching
  `magic.MagicException` and re-raising as `IngestError`.

- [x] **api.py ~line 231** — write failure orphaned staging file (covered by fix above).

- [x] **config.py:110** — `ADMIN_PASSWORD_HASH` not validated as bcrypt hash.
  **Fixed:** Added `bcrypt.checkpw(b"probe", ADMIN_PASSWORD_HASH.encode())` probe in
  `validate_server()`; `ValueError` → `SystemExit` with a clear message.

---

## Medium (data integrity / reliability)

- [x] **ingest.py:181** — `CancelledError` bypassed both `except` handlers.
  **Fixed:** Added `except BaseException` block that sets status to `"failed"` and re-raises.

- [x] **ui/static/app.js:145** — `loadHistory` no `resp.ok` check.
  **Fixed:** Added `if (!resp.ok) { if (resp.status === 401) window.location.href = '/login'; return; }`

- [x] **api.py:274** — `post_login` discards CSRF token with no explanation.
  **Fixed:** Added inline comment: CSRF token is embedded in `GET /` HTML, not the redirect.

- [x] **db.py:37** — `executescript()` implicit COMMIT broke rollback guarantee.
  **Fixed:** Replaced with two separate `conn.execute()` calls inside the `_db()` transaction.

---

## Low (display / memory)

- [x] **ui/static/app.js:108** — `escHtml()` output passed to `textContent` → literal entities.
  **Fixed:** Removed `escHtml()` from the two `showStatus` callsites in `pollJob`; `textContent`
  provides the escaping.

- [x] **api.py:92** — rate-limit dicts grew unboundedly.
  **Fixed:** Switched from `defaultdict` to plain `dict`; `dispatch` evicts the oldest entry
  (insertion-order) when size reaches `_MAX_BUCKET_SIZE = 10_000`.

---

## Deferred (by explicit user instruction)

- Three-instance Ollama routing (OLLAMA_PLANNER_URL / OLLAMA_RESEARCHER_URL / OLLAMA_EMBED_URL
  at ports 11434/11435/11436) — wire at project end.
- RAG ingest endpoint signatures (RAG_INGEST_URL, GRAPH_RAG_INGEST_URL) — confirm with
  rag-system and local-graph-rag before wiring.
