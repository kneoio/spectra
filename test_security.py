import importlib
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

ISSUER = "https://auth.semantyca.com/realms/mixpla"


@pytest.fixture(scope="module")
def rsa_keys():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _token(private_pem, sub="user-1", issuer=ISSUER, expires_in=300):
    now = int(time.time())
    return jwt.encode(
        {"sub": sub, "iss": issuer, "iat": now, "exp": now + expires_in},
        private_pem,
        algorithm="RS256",
    )


def build(monkeypatch, public_pem=None, **env):
    for key in ("SPECTRA_API_KEY", "SPECTRA_OIDC_ISSUER", "SPECTRA_OIDC_AUDIENCE",
                "SPECTRA_RATE_LIMIT_PER_MINUTE", "SPECTRA_MAX_CONCURRENT_MIXES", "SPECTRA_ANALYZE_ROOTS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import security
    security = importlib.reload(security)
    if public_pem is not None:
        monkeypatch.setattr(security, "_get_signing_key", lambda token: public_pem)
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/internal", dependencies=[Depends(security.require_internal_key)])
    def internal():
        return {"ok": True}

    @app.post("/heavy", dependencies=[Depends(security.require_mix_access), Depends(security.rate_limit), Depends(security.mix_slot)])
    def heavy():
        return {"ok": True}

    return security, TestClient(app)


def test_health_is_always_open(monkeypatch):
    _, client = build(monkeypatch, SPECTRA_API_KEY="secret")
    assert client.get("/health").status_code == 200


def test_internal_key_required_for_internal_routes(monkeypatch):
    _, client = build(monkeypatch, SPECTRA_API_KEY="secret")
    assert client.post("/internal").status_code == 401
    assert client.post("/internal", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/internal", headers={"X-API-Key": "secret"}).status_code == 200


def test_open_when_no_key_configured(monkeypatch):
    _, client = build(monkeypatch)
    assert client.post("/internal").status_code == 200
    assert client.post("/heavy").status_code == 200


def test_mix_route_requires_credentials_when_oidc_configured(monkeypatch):
    _, client = build(monkeypatch, SPECTRA_OIDC_ISSUER=ISSUER)
    assert client.post("/heavy").status_code == 401


def test_mix_route_accepts_a_valid_player_bearer_token(monkeypatch, rsa_keys):
    private_pem, public_pem = rsa_keys
    _, client = build(monkeypatch, public_pem=public_pem, SPECTRA_OIDC_ISSUER=ISSUER)
    token = _token(private_pem)
    assert client.post("/heavy", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_mix_route_rejects_an_expired_or_wrong_issuer_token(monkeypatch, rsa_keys):
    private_pem, public_pem = rsa_keys
    _, client = build(monkeypatch, public_pem=public_pem, SPECTRA_OIDC_ISSUER=ISSUER)
    expired = _token(private_pem, expires_in=-10)
    assert client.post("/heavy", headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    wrong_issuer = _token(private_pem, issuer="https://evil.example/realms/other")
    assert client.post("/heavy", headers={"Authorization": f"Bearer {wrong_issuer}"}).status_code == 401


def test_mix_route_still_accepts_the_internal_key_as_fallback(monkeypatch, rsa_keys):
    _, public_pem = rsa_keys
    _, client = build(monkeypatch, public_pem=public_pem, SPECTRA_OIDC_ISSUER=ISSUER, SPECTRA_API_KEY="secret")
    assert client.post("/heavy", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.post("/heavy", headers={"X-API-Key": "wrong"}).status_code == 401


def test_each_player_gets_its_own_rate_limit_bucket(monkeypatch, rsa_keys):
    private_pem, public_pem = rsa_keys
    _, client = build(monkeypatch, public_pem=public_pem, SPECTRA_OIDC_ISSUER=ISSUER,
                       SPECTRA_RATE_LIMIT_PER_MINUTE="1")
    token_a = _token(private_pem, sub="user-a")
    token_b = _token(private_pem, sub="user-b")
    assert client.post("/heavy", headers={"Authorization": f"Bearer {token_a}"}).status_code == 200
    assert client.post("/heavy", headers={"Authorization": f"Bearer {token_a}"}).status_code == 429
    # a second player's own token is a separate bucket, not blocked by user-a's hit
    assert client.post("/heavy", headers={"Authorization": f"Bearer {token_b}"}).status_code == 200


def test_rate_limit(monkeypatch):
    _, client = build(monkeypatch, SPECTRA_RATE_LIMIT_PER_MINUTE="3")
    assert [client.post("/heavy").status_code for _ in range(3)] == [200, 200, 200]
    blocked = client.post("/heavy")
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1


def test_concurrent_mix_limit(monkeypatch):
    security, client = build(monkeypatch, SPECTRA_MAX_CONCURRENT_MIXES="1")
    security.acquire_mix_slot()
    assert client.post("/heavy").status_code == 429
    security.release_mix_slot()
    assert client.post("/heavy").status_code == 200


def test_analyze_path_restricted_to_roots(monkeypatch, tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    inside = media / "song.mp3"
    inside.write_bytes(b"x")
    outside = tmp_path / "secret.txt"
    outside.write_bytes(b"x")
    security, _ = build(monkeypatch, SPECTRA_ANALYZE_ROOTS=str(media))
    assert security.allowed_analyze_path(str(inside)) == str(inside)
    assert security.allowed_analyze_path(str(outside)) is None
    assert security.allowed_analyze_path(str(media / ".." / "secret.txt")) is None


def test_analyze_path_ignored_without_roots(monkeypatch, tmp_path):
    file = tmp_path / "song.mp3"
    file.write_bytes(b"x")
    security, _ = build(monkeypatch)
    assert security.allowed_analyze_path(str(file)) is None
