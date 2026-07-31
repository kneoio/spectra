import logging
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from analysis import analyze, get_model_bundle, is_music
from db import get_original_file_key, save_analysis
from models_setup import ensure_models
from storage import download_to_temp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("spectra.service")


class AnalyzeRequest(BaseModel):
    soundFragmentId: str
    path: str | None = None      # local original on disk (optimization); resolved
                                 # from _files when absent/missing


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_models(quiet=True)
    get_model_bundle()  # build all TF graphs once, kept resident for the process lifetime
    yield


app = FastAPI(title="Spectra", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/assess")
async def assess_track(file: UploadFile = File(...)) -> dict:
    """Synchronous pre-save assessment: upload a local audio file, get analysis
    back in the response (including is_music). Writes nothing to the database —
    used by jesoos chat (assess_track / upload_song) before a SoundFragment exists.
    """
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
        result = await run_in_threadpool(analyze, tmp_path)
        result.pop("file", None)
        if "duration_sec" in result:
            result["duration_seconds"] = result["duration_sec"]
        result["is_music"] = is_music(result)
        logger.info(
            "Assessed %s is_music=%s duration=%s bpm=%s",
            file.filename, result["is_music"], result.get("duration_sec"), result.get("bpm"),
        )
        return result
    except Exception as e:
        logger.exception("Assess failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _resolve_file(sound_fragment_id: str, path: str | None) -> tuple[str, bool]:
    """Prefer the local original if it's still on disk (optimization); otherwise
    materialize the ORIGINAL (non-opus) file from Hetzner, resolving its key from
    _files by the SoundFragment id. Returns (local_path, is_temp) — is_temp marks
    a downloaded copy spectra must delete."""
    if path and os.path.isfile(path):
        return path, False
    file_key = get_original_file_key(sound_fragment_id)
    if not file_key:
        raise FileNotFoundError(
            f"No local path {path!r} and no original file in _files for SF={sound_fragment_id}"
        )
    return download_to_temp(file_key), True


def _analyze_and_store(sound_fragment_id: str, path: str | None) -> None:
    local_path, is_temp = None, False
    try:
        local_path, is_temp = _resolve_file(sound_fragment_id, path)
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
    background_tasks.add_task(_analyze_and_store, request.soundFragmentId, request.path)
    return {"status": "accepted", "soundFragmentId": request.soundFragmentId}
