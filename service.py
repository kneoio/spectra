import logging
import os
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from analysis import analyze, get_model_bundle
from db import save_analysis
from models_setup import ensure_models
from storage import download_to_temp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("spectra.service")


class AnalyzeRequest(BaseModel):
    soundFragmentId: str
    path: str | None = None      # local original, if still on disk (shared volume)
    fileKey: str | None = None   # Hetzner object key, used as fallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_models(quiet=True)
    get_model_bundle()  # build all TF graphs once, kept resident for the process lifetime
    yield


app = FastAPI(title="Spectra", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


def _resolve_file(path: str | None, file_key: str | None) -> tuple[str, bool]:
    """Prefer the local original if it's still on disk (shared volume, not yet
    cleaned); otherwise download the original from Hetzner by fileKey.
    Returns (local_path, is_temp) — is_temp marks a copy spectra must delete."""
    if path and os.path.isfile(path):
        return path, False
    if file_key:
        return download_to_temp(file_key), True
    raise FileNotFoundError(f"No local file at {path!r} and no fileKey provided")


def _analyze_and_store(sound_fragment_id: str, path: str | None, file_key: str | None) -> None:
    local_path, is_temp = None, False
    try:
        local_path, is_temp = _resolve_file(path, file_key)
        result = analyze(local_path)
        result.pop("file", None)  # local/temp path is not meaningful to persist
        updated = save_analysis(sound_fragment_id, result)
        if updated:
            logger.info("Analyzed and stored SF=%s (temp=%s)", sound_fragment_id, is_temp)
        else:
            logger.warning("SF=%s not found in DB; analysis discarded", sound_fragment_id)
    except Exception:
        logger.exception("Analysis failed for SF=%s", sound_fragment_id)
    finally:
        # Only delete copies spectra downloaded itself; never the shared local
        # original (that belongs to datanest / metriq cleanup).
        if is_temp and local_path and os.path.exists(local_path):
            os.remove(local_path)


@app.post("/analyze", status_code=202)
def analyze_track(request: AnalyzeRequest, background_tasks: BackgroundTasks) -> dict:
    """Accept a file reference over REST, acknowledge immediately (202), and run
    analysis in the background. On success the result is written to the
    SoundFragment's `add_info` column — it does NOT come back in the HTTP
    response."""
    background_tasks.add_task(_analyze_and_store, request.soundFragmentId, request.path, request.fileKey)
    return {"status": "accepted", "soundFragmentId": request.soundFragmentId}
