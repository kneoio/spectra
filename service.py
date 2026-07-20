import os
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from analysis import analyze, get_model_bundle
from messaging import open_channel, publish_metric
from models_setup import ensure_models


class AnalyzeRequest(BaseModel):
    path: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_models(quiet=True)
    get_model_bundle()  # build all TF graphs once, kept resident for the process lifetime
    yield


app = FastAPI(title="Spectra", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


def _analyze_and_publish(path: str) -> None:
    try:
        result = analyze(path)
        event_type, code, payload = "INFORMATION", "track_analyzed", {"file": path, "metadata": result}
    except RuntimeError as e:
        event_type, code, payload = "ERROR", "analysis_failed", {"file": path, "error": str(e)}

    try:
        connection, channel = open_channel()
        try:
            publish_metric(channel, event_type=event_type, process_type="FLOW", code=code, payload=payload)
        finally:
            connection.close()
    except Exception:
        pass  # metrics-pipeline outage must not crash the background task


@app.post("/analyze", status_code=202)
def analyze_track(request: AnalyzeRequest, background_tasks: BackgroundTasks) -> dict:
    """Accepts a file path over REST, runs analysis in the background, and
    publishes the result as a `track_analyzed`/`analysis_failed` metric event
    on the shared `metrics` exchange — the result does NOT come back in the
    HTTP response, only an acknowledgement that analysis was queued."""
    path = request.path
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    background_tasks.add_task(_analyze_and_publish, path)
    return {"status": "accepted", "file": path}
