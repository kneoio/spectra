import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from mixer import plan_mix
from render import render as render_mix
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from analysis import analyze, get_model_bundle, is_music, spectral_map
from db import get_original_file_key, save_analysis
from models_setup import ensure_models
from storage import download_to_temp

DEFAULT_MIX_SECONDS = 30.0
MIX_JOB_TTL_SECONDS = 1800
MIX_JOB_REAP_INTERVAL_SECONDS = 300

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("spectra.service")


class AnalyzeRequest(BaseModel):
    soundFragmentId: str
    path: str | None = None      # local original on disk (optimization); resolved
                                 # from _files when absent/missing


@dataclass
class MixJob:
    """In-memory state for one /mix/jobs run: a queue of progress events an
    SSE stream reads from, plus the eventual result (or error)."""
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    status: str = "running"  # running, done, error
    created_at: float = field(default_factory=time.monotonic)
    result_path: str | None = None
    plan: dict | None = None
    error: str | None = None


_mix_jobs: dict[str, MixJob] = {}


async def _reap_mix_jobs() -> None:
    """Drop finished/abandoned jobs (and their result files) after
    MIX_JOB_TTL_SECONDS — a client that never calls /result would otherwise
    leak a rendered WAV per job forever."""
    while True:
        await asyncio.sleep(MIX_JOB_REAP_INTERVAL_SECONDS)
        cutoff = time.monotonic() - MIX_JOB_TTL_SECONDS
        stale = [job_id for job_id, job in _mix_jobs.items() if job.created_at < cutoff]
        for job_id in stale:
            job = _mix_jobs.pop(job_id, None)
            if job:
                _cleanup(job.result_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_models(quiet=True)
    get_model_bundle()  # build all TF graphs once, kept resident for the process lifetime
    reaper = asyncio.create_task(_reap_mix_jobs())
    yield
    reaper.cancel()


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


@app.post("/spectral-map")
async def spectral_map_track(
    file: UploadFile = File(...),
    segment: str = Form(...),
    seconds: float = Form(30.0),
) -> dict:
    """Band-energy/rhythm time series for one edge (head/tail) of an uploaded
    track — used by mixer.plan_mix to plan a beat-aligned crossfade. Writes
    nothing to the database."""
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
        return await run_in_threadpool(spectral_map, tmp_path, segment, seconds)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Spectral map failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


async def _save_upload(file: UploadFile) -> str:
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return tmp_path


def _cleanup(*paths: str | None) -> None:
    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)


@app.post("/mix")
async def mix_tracks(
    file_a: UploadFile = File(..., description="outgoing track (tail is analyzed)"),
    file_c: UploadFile = File(..., description="incoming track (head is analyzed)"),
    seconds: float = Form(DEFAULT_MIX_SECONDS),
    keylock: bool = Form(True),
) -> FileResponse:
    """Analyze A's tail and C's head, plan a beat-aligned crossfade
    (mixer.plan_mix), render it (render.render), and return the mixed WAV.
    The plan itself comes back as the X-Mix-Plan header."""
    tmp_a = tmp_c = tmp_out = None
    try:
        tmp_a = await _save_upload(file_a)
        tmp_c = await _save_upload(file_c)
        map_a = await run_in_threadpool(spectral_map, tmp_a, "tail", seconds)
        map_c = await run_in_threadpool(spectral_map, tmp_c, "head", seconds)
        plan = await run_in_threadpool(plan_mix, map_a, map_c)

        fd, tmp_out = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        await run_in_threadpool(render_mix, plan, tmp_a, tmp_c, tmp_out, keylock)

        return FileResponse(
            tmp_out,
            media_type="audio/wav",
            filename="mix.wav",
            headers={"X-Mix-Plan": json.dumps(plan)},
            background=BackgroundTask(_cleanup, tmp_a, tmp_c, tmp_out),
        )
    except (ValueError, RuntimeError) as e:
        _cleanup(tmp_a, tmp_c, tmp_out)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Mix failed for %s / %s", file_a.filename, file_c.filename)
        _cleanup(tmp_a, tmp_c, tmp_out)
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _emit(job: MixJob, job_id: str, name: str, status: str, error_message: str | None = None) -> None:
    """SSEProgressDTO-shaped event (id/name/status/errorMessage) — matches the
    com.semantyca.core / io.kneo.broadcaster SSE progress convention so aivox
    can deserialize it directly. status is PROCESSING, DONE or ERROR."""
    await job.queue.put({"id": job_id, "name": name, "status": status, "errorMessage": error_message})


async def _run_mix_job(job: MixJob, job_id: str, tmp_a: str, tmp_c: str, seconds: float, keylock: bool) -> None:
    tmp_out = None
    try:
        await _emit(job, job_id, "Analyzing outgoing track's tail", "PROCESSING")
        map_a = await run_in_threadpool(spectral_map, tmp_a, "tail", seconds)
        await _emit(job, job_id, "Analyzing incoming track's head", "PROCESSING")
        map_c = await run_in_threadpool(spectral_map, tmp_c, "head", seconds)
        await _emit(job, job_id, "Planning the crossfade", "PROCESSING")
        plan = await run_in_threadpool(plan_mix, map_a, map_c)
        job.plan = plan
        await _emit(job, job_id, "Rendering the mix", "PROCESSING")
        fd, tmp_out = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        await run_in_threadpool(render_mix, plan, tmp_a, tmp_c, tmp_out, keylock)
        job.result_path = tmp_out
        job.status = "done"
        await _emit(job, job_id, "Mix ready", "DONE")
    except (ValueError, RuntimeError) as e:
        job.status = "error"
        job.error = str(e)
        _cleanup(tmp_out)
        await _emit(job, job_id, "Mix failed", "ERROR", str(e))
    except Exception as e:
        logger.exception("Mix job failed for %s / %s", tmp_a, tmp_c)
        job.status = "error"
        job.error = str(e)
        _cleanup(tmp_out)
        await _emit(job, job_id, "Mix failed", "ERROR", str(e))
    finally:
        _cleanup(tmp_a, tmp_c)


@app.post("/mix/jobs", status_code=202)
async def create_mix_job(
    file_a: UploadFile = File(..., description="outgoing track (tail is analyzed)"),
    file_c: UploadFile = File(..., description="incoming track (head is analyzed)"),
    seconds: float = Form(DEFAULT_MIX_SECONDS),
    keylock: bool = Form(True),
) -> dict:
    """Start a /mix run in the background and return a job_id immediately.
    Progress streams from GET /mix/jobs/{job_id}/events (SSE); the finished
    WAV is fetched from GET /mix/jobs/{job_id}/result."""
    tmp_a = await _save_upload(file_a)
    tmp_c = await _save_upload(file_c)
    job_id = uuid.uuid4().hex
    job = MixJob()
    _mix_jobs[job_id] = job
    asyncio.create_task(_run_mix_job(job, job_id, tmp_a, tmp_c, seconds, keylock))
    return {"job_id": job_id}


@app.get("/mix/jobs/{job_id}/events")
async def mix_job_events(job_id: str) -> StreamingResponse:
    job = _mix_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def event_stream():
        while True:
            event = await job.queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["status"] in ("DONE", "ERROR"):
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/mix/jobs/{job_id}/result")
async def mix_job_result(job_id: str) -> FileResponse:
    job = _mix_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status == "error":
        _mix_jobs.pop(job_id, None)
        raise HTTPException(status_code=400, detail=job.error)
    if job.status != "done" or not job.result_path:
        raise HTTPException(status_code=409, detail="job not finished yet")

    # Pop immediately so a duplicate/concurrent fetch gets a clean 404 rather
    # than racing the background file delete below.
    _mix_jobs.pop(job_id, None)

    return FileResponse(
        job.result_path,
        media_type="audio/wav",
        filename="mix.wav",
        headers={"X-Mix-Plan": json.dumps(job.plan)},
        background=BackgroundTask(_cleanup, job.result_path),
    )


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
