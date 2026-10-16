import hmac
import logging
import os
import time
from collections import deque

from fastapi import HTTPException, Request

logger = logging.getLogger("spectra.security")

API_KEY = os.environ.get("SPECTRA_API_KEY", "")
RATE_LIMIT_PER_MINUTE = int(os.environ.get("SPECTRA_RATE_LIMIT_PER_MINUTE", "30"))
MAX_CONCURRENT_MIXES = int(os.environ.get("SPECTRA_MAX_CONCURRENT_MIXES", "2"))
ANALYZE_ROOTS = [os.path.realpath(p) for p in os.environ.get("SPECTRA_ANALYZE_ROOTS", "").split(os.pathsep) if p]
RATE_WINDOW_SECONDS = 60
PUBLIC_PATHS = {"/health"}

_hits: dict[str, deque[float]] = {}
_active_mixes = 0


def warn_if_open() -> None:
    if not API_KEY:
        logger.warning("SPECTRA_API_KEY is not set: authentication is disabled")


def require_api_key(request: Request) -> None:
    if not API_KEY or request.url.path in PUBLIC_PATHS:
        return
    supplied = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(supplied.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _client_id(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    now = time.monotonic()
    hits = _hits.setdefault(_client_id(request), deque())
    while hits and now - hits[0] > RATE_WINDOW_SECONDS:
        hits.popleft()
    if len(hits) >= RATE_LIMIT_PER_MINUTE:
        retry = int(RATE_WINDOW_SECONDS - (now - hits[0])) + 1
        raise HTTPException(status_code=429, detail="rate limit exceeded", headers={"Retry-After": str(retry)})
    hits.append(now)


def prune_rate_limits() -> None:
    cutoff = time.monotonic() - RATE_WINDOW_SECONDS
    for client in [c for c, hits in _hits.items() if not hits or hits[-1] < cutoff]:
        _hits.pop(client, None)


def acquire_mix_slot() -> None:
    global _active_mixes
    if _active_mixes >= MAX_CONCURRENT_MIXES:
        raise HTTPException(status_code=429, detail="server is busy mixing, retry shortly", headers={"Retry-After": "10"})
    _active_mixes += 1


def release_mix_slot() -> None:
    global _active_mixes
    _active_mixes = max(0, _active_mixes - 1)


async def mix_slot():
    acquire_mix_slot()
    try:
        yield
    finally:
        release_mix_slot()


def allowed_analyze_path(path: str | None) -> str | None:
    if not path or not ANALYZE_ROOTS:
        return None
    real = os.path.realpath(path)
    if any(real == root or real.startswith(root + os.sep) for root in ANALYZE_ROOTS):
        return real
    return None
