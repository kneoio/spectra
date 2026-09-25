FROM python:3.14-slim

# ffmpeg supplies the `ffprobe` binary used by the AI-metadata scan, plus audio
# codecs; curl is used by ensure_models() as the model-download fallback;
# rubberband-cli backs render.py's keylock time-stretch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl ca-certificates rubberband-cli \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer — cached unless the lockfile changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# App source + baked-in TF models (~23 MB). Any missing model is fetched at
# startup by ensure_models() as a fallback.
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 38795

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "38795"]
