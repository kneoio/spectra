import hmac
import logging
import os
import time
from collections import deque

import jwt
from fastapi import HTTPException, Request

logger = logging.getLogger("spectra.security")

API_KEY = os.environ.get("SPECTRA_API_KEY", "")

# Each Android player authenticates its own user against Mixpla's existing
# Keycloak realm and sends that access token straight through as a Bearer
# token — Spectra verifies it against the same realm rather than issuing or
# storing any credential of its own. Unset (the default) disables Bearer
# auth entirely, e.g. for local dev.
OIDC_ISSUER = os.environ.get("SPECTRA_OIDC_ISSUER", "")
OIDC_AUDIENCE = os.environ.get("SPECTRA_OIDC_AUDIENCE", "")
OIDC_JWKS_URL = os.environ.get(
    "SPECTRA_OIDC_JWKS_URL",
    f"{OIDC_ISSUER}/protocol/openid-connect/certs" if OIDC_ISSUER else "",
)

RATE_LIMIT_PER_MINUTE = int(os.environ.get("SPECTRA_RATE_LIMIT_PER_MINUTE", "30"))
MAX_CONCURRENT_MIXES = int(os.environ.get("SPECTRA_MAX_CONCURRENT_MIXES", "2"))
ANALYZE_ROOTS = [os.path.realpath(p) for p in os.environ.get("SPECTRA_ANALYZE_ROOTS", "").split(os.pathsep) if p]
RATE_WINDOW_SECONDS = 60

_hits: dict[str, deque[float]] = {}
_active_mixes = 0
_jwks_client: jwt.PyJWKClient | None = None


def warn_if_open() -> None:
    if not API_KEY:
        logger.warning("SPECTRA_API_KEY is not set: the Mixpla-internal endpoints have no authentication")
    if not OIDC_ISSUER:
        logger.warning("SPECTRA_OIDC_ISSUER is not set: the public mix endpoints accept no player Bearer tokens")


def require_internal_key(request: Request) -> None:
    """Gate for the Mixpla-internal, DB-coupled endpoints (/analyze, /assess,
    /spectral-map). Only the service's own key is accepted here — a player's
    user token must never reach the database-backed routes."""
    if not API_KEY:
        return
    supplied = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(supplied.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _get_signing_key(token: str) -> str:
    """Looked up separately from _verify_user_token so tests can substitute a
    fixed key instead of fetching Keycloak's real JWKS over the network."""
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = jwt.PyJWKClient(OIDC_JWKS_URL)
    return _jwks_client.get_signing_key_from_jwt(token).key


def _verify_user_token(token: str) -> dict:
    signing_key = _get_signing_key(token)
    return jwt.decode(
        token,
        signing_key,
        algorithms=["RS256"],
        issuer=OIDC_ISSUER,
        audience=OIDC_AUDIENCE or None,
        options={"verify_aud": bool(OIDC_AUDIENCE)},
    )


def require_mix_access(request: Request) -> None:
    """Gate for the public mixing endpoints (/mix, /mix/analyze, /mix/jobs...).
    Each player sends its own user's Keycloak access token as a Bearer token
    — a leaked token only ever exposes that one user's account, and Keycloak
    (expiry/revocation) is the only place credentials need managing. Mixpla's
    own server-to-server calls can still use the internal API key instead.
    Sets request.state.client_id to the caller's identity so rate_limit can
    give each player its own quota rather than bucketing by IP."""
    if not API_KEY and not OIDC_ISSUER:
        return  # nothing configured at all: open, e.g. local dev

    auth = request.headers.get("authorization", "")
    if OIDC_ISSUER and auth.lower().startswith("bearer "):
        token = auth[len("bearer "):].strip()
        try:
            claims = _verify_user_token(token)
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"invalid or expired token: {e}") from e
        request.state.client_id = f"user:{claims.get('sub', 'unknown')}"
        return

    supplied = request.headers.get("x-api-key", "")
    if API_KEY and hmac.compare_digest(supplied.encode(), API_KEY.encode()):
        request.state.client_id = "internal"
        return

    raise HTTPException(status_code=401, detail="missing or invalid credentials (Bearer token or X-API-Key)")


def _client_id(request: Request) -> str:
    client_id = getattr(request.state, "client_id", None)
    if client_id:
        return client_id
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
