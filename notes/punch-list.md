# Code Review Punch List

Found by `/code-review` on 2026-06-26. All open, none fixed yet.

---

## Critical (security / won't start)

- [ ] **Dockerfile + config.py** — `CMD --host 0.0.0.0` is disconnected from `config.HOST`
  (defaults to `127.0.0.1`). `validate_server()` passes the loopback check, but uvicorn
  actually binds all interfaces. With `ALLOW_INSECURE_LOCALONLY=true` and no `API_KEY`, every
  `/v1/*` route is open to the network unauthenticated.
  **Fix:** Pass `HOST` env var into the CMD (`--host ${HOST}`), or always pass `--host 0.0.0.0`
  and remove the loopback binding check from `validate_server()` in favour of a network-level
  firewall note in the docs. Simplest: use `uvicorn ... --host ${HOST:-0.0.0.0}` in a shell
  entrypoint and keep the validation.

- [ ] **docker-compose.yml:10** — `HOST: "0.0.0.0"` causes `validate_server()` to
  `SystemExit` when `ALLOW_INSECURE_LOCALONLY=true` (the default in `.env.example`), so
  `docker-compose up` never starts.
  **Fix:** Remove `HOST` from the compose environment block and rely on the CMD `--host 0.0.0.0`
  flag (or use a shell entrypoint that reads `HOST` from the env file).

---

## High (wrong behaviour / auth)

- [ ] **api.py ~line 294** — `resp.delete_cookie(_SESSION_COOKIE)` is called without
  `secure=True`. Chrome refuses to clear a `Secure`-flagged cookie via a non-Secure response.
  Session cookie persists in the browser after logout on HTTPS deployments.
  **Fix:** `resp.delete_cookie(_SESSION_COOKIE, secure=_cookie_secure(request))`

- [ ] **api.py ~line 184** — `provided = auth[len("Bearer "):]` is not stripped.
  A trailing space in the `Authorization` header produces a false 401 for a correct key.
  **Fix:** `provided = auth[len("Bearer "):].strip()`

- [ ] **api.py ~line 234** — `ingest.validate_upload(dest, staging)` is a synchronous call
  inside an `async` function. It runs `is_symlink()`, `resolve()`, `stat()`, and `read_bytes()`
  (via `_check_mime`) on the event loop thread, blocking all concurrent requests.
  **Fix:** `await asyncio.to_thread(ingest.validate_upload, dest, staging)`

- [ ] **ingest.py:172** — `filename = path.name` captures the `secrets.token_hex(8)_` prefix
  added in `_accept_upload`, so RAG ingest services receive `a1b2c3d4_report.pdf` instead of
  `report.pdf`.
  **Fix:** Pass the original filename into `run_background_indexing` as a separate argument
  instead of deriving it from `path.name`.

- [ ] **ingest.py:84** — `except ImportError` does not catch `magic.MagicException`, which
  is raised when the `python-magic` package is installed but the native `libmagic` shared
  library is absent. Results in a 500 instead of a clean 422.
  **Fix:** `except (ImportError, Exception)` where the inner `Exception` branch re-raises
  non-MagicException errors, or catch `magic.MagicException` explicitly after the `import magic`.

- [ ] **api.py ~line 231** — If `asyncio.to_thread(dest.write_bytes, contents)` raises
  (disk full, EPERM), the partially-created staging file is never cleaned up — the `try/except`
  that calls `dest.unlink()` only starts after the write succeeds.
  **Fix:** Wrap the write in the same `try` block, or use a `try/finally` around both write and
  validate steps.

- [ ] **config.py:110** — `validate_server()` checks `ADMIN_PASSWORD_HASH` is non-empty but
  not that it is a valid bcrypt hash. An invalid value (e.g. `changeme`) passes startup;
  `bcrypt.checkpw` raises `ValueError('Invalid salt')`, caught by `except Exception: valid = False`,
  silently locking out every login with no diagnostic.
  **Fix:** Add `bcrypt.checkpw(b'probe', config.ADMIN_PASSWORD_HASH.encode())` in a try/except
  in `validate_server()`, or at minimum check that the value starts with `$2b$`.

---

## Medium (data integrity / reliability)

- [ ] **ingest.py:181** — `asyncio.CancelledError` is a `BaseException` (Python 3.8+) and
  bypasses both `except IngestError` and `except Exception`. On server shutdown, a mid-flight
  indexing task is cancelled and the job is permanently stuck at `'indexing'`.
  **Fix:** Add `except BaseException: _jobs[job_id]["status"] = "failed"; _jobs[job_id]["error"] = "Cancelled"; raise`

- [ ] **ui/static/app.js:145** — `loadHistory` calls `resp.json()` without checking
  `resp.ok` first. A 401 body parses as an object; `.map()` throws `TypeError`; `catch {}`
  swallows it silently — user gets no indication their session has expired.
  **Fix:** Add `if (!resp.ok) { /* redirect or show error */ return; }` before `resp.json()`.

- [ ] **api.py:274** — `post_login` discards the CSRF token (`token, _csrf = db.create_session()`).
  Non-browser clients that obtain a session cookie via `POST /login` and immediately call
  `/ui/*` endpoints will always get 403 — they have no way to get the CSRF token without
  parsing the HTML of `GET /`. Intended non-browser path is `/v1/*` (bearer token).
  **Fix:** Document clearly in README that `/ui/*` is browser-only; or return CSRF token in
  a response header on the `POST /login` redirect.

- [ ] **db.py:37** — `executescript()` issues an implicit `COMMIT` before running DDL,
  breaking `_db()`'s rollback guarantee. If the `sessions` table creation fails mid-script,
  the `queries` table is already committed and `conn.rollback()` is a no-op.
  **Fix:** Replace `executescript` with two separate `conn.execute("CREATE TABLE IF NOT EXISTS ...")`
  calls inside the existing `_db()` transaction so rollback works correctly.

---

## Low (display / memory)

- [ ] **ui/static/app.js:108** — `showStatus` sets `el.textContent` (which is XSS-safe) but
  callers pass strings already processed by `escHtml()`, so entities render literally
  (`Q&amp;A` instead of `Q&A`).
  **Fix:** Remove `escHtml()` from the two `showStatus` callsites in `pollJob` — `textContent`
  provides the escaping.

- [ ] **api.py:92** — `_buckets` and `_login_buckets` are `defaultdict`s with no eviction.
  Every unique source IP creates a permanent `_Bucket` entry. Under rotating IPs or spoofed XFF,
  the dicts grow indefinitely, causing monotonic RSS growth.
  **Fix:** Cap with an LRU structure (e.g. `functools.lru_cache` keyed on IP, or a
  `collections.OrderedDict` with a max size), or add a periodic sweep that removes buckets
  whose `tokens` have been at capacity for longer than one window.
