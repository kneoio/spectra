import importlib
import os

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient


def build(monkeypatch, **env):
    for key in ("SPECTRA_API_KEY", "SPECTRA_RATE_LIMIT_PER_MINUTE", "SPECTRA_MAX_CONCURRENT_MIXES", "SPECTRA_ANALYZE_ROOTS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import security
    security = importlib.reload(security)
    app = FastAPI(dependencies=[Depends(security.require_api_key)])

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/heavy", dependencies=[Depends(security.rate_limit), Depends(security.mix_slot)])
    def heavy():
        return {"ok": True}

    return security, TestClient(app)


def test_api_key_required_except_health(monkeypatch):
    _, client = build(monkeypatch, SPECTRA_API_KEY="secret")
    assert client.get("/health").status_code == 200
    assert client.post("/heavy").status_code == 401
    assert client.post("/heavy", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/heavy", headers={"X-API-Key": "secret"}).status_code == 200


def test_open_when_no_key_configured(monkeypatch):
    _, client = build(monkeypatch)
    assert client.post("/heavy").status_code == 200


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
