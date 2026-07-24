# spectra

Audio-analysis service for mixpla. Given a SoundFragment, it extracts musical
features (BPM, key/scale, loudness, genre, moods, danceability) plus a weak
AI-generation metadata check, and writes the result onto the SoundFragment row.

Analysis runs on [Essentia](https://essentia.upf.edu/) with the Discogs-EffNet
TensorFlow models. It's a FastAPI service on the Compose host network at
`:38795` — internal only, no public route.

## Flow

```
datanest (on save) ──POST /analyze {soundFragmentId, path?}──▶ spectra
                                                                  │
                          resolve file: local `path` if on disk,  │
                          else original file_key from `_files`     │
                          (opus excluded) ──▶ download from Hetzner │
                                                                  ▼
                                    Essentia + TF analysis
                                                                  ▼
                    UPDATE mixpla__sound_fragments.add_info (moon DB)
```

The result is **not** returned in the HTTP response — the endpoint acknowledges
with `202` and analysis completes asynchronously, persisting to the database.

## API

### `POST /analyze`
Accepts a file reference, acknowledges immediately, analyzes in the background.

```json
{ "soundFragmentId": "<uuid>", "path": "/optional/local/original.wav" }
```

- `soundFragmentId` (required) — the SoundFragment to analyze and store onto.
- `path` (optional) — a local original on disk. Used only if it exists; it's a
  fast-path optimization that skips the download.

Response `202`:
```json
{ "status": "accepted", "soundFragmentId": "<uuid>" }
```

**File resolution:** `path` is used if present on disk. Otherwise spectra
resolves the **original** (non-opus) `file_key` from `_files` by `parent_id`
(`file_type <> 102`, preferring `101` over legacy `0`) and downloads it from
Hetzner object storage. Downloaded copies are temp files, deleted immediately
after analysis; the shared local original (if used) is never deleted.

The result is written to `mixpla__sound_fragments.add_info` (jsonb), e.g.:
```json
{
  "bpm": 101.79, "key": "Ab", "scale": "minor", "loudness": 3205.87,
  "moods": {"happy": 0.304, "sad": 0.121, "party": 0.514, "relaxed": 0.659, "aggressive": 0.283},
  "danceability": 0.71,
  "top_genres": [{"genre": "Electronic---Berlin-School", "score": 0.227}],
  "ai_generated_metadata_check": {"suspected_ai_generated": false, "evidence": []}
}
```

### `GET /health`
```json
{ "status": "ok" }
```

## Configuration (env)

| Variable | Purpose | Default |
|---|---|---|
| `SPECTRA_DB_HOST` / `SPECTRA_DB_PORT` | moon DB | `127.0.0.1` / `8572` |
| `SPECTRA_DB_NAME` / `SPECTRA_DB_USER` / `SPECTRA_DB_PASSWORD` | moon DB | `moon` / `regolith` / — |
| `HETZNER_STORAGE_ENDPOINT` / `HETZNER_STORAGE_BUCKET` | object storage | `https://hel1.your-objectstorage.com` / `soundfragments` |
| `HETZNER_STORAGE_ACCESS_KEY` / `HETZNER_STORAGE_SECRET_KEY` | object storage | — |

On the server these live in `~/compose/env/spectra.env`.

## Run locally

```bash
uv sync
uv run python models_setup.py      # download the TF models (~23 MB) once
uv run uvicorn service:app --host 0.0.0.0 --port 38795
```

`main.py` runs a one-off analysis of a single file for quick checks.

## Deployment

Containerized (`Dockerfile`, python 3.14-slim + ffmpeg). CI builds and pushes
`ghcr.io/kneoio/spectra:latest` via the manual **Build and Push** workflow
(`.github/workflows/deploy.yml`, `workflow_dispatch`).

On the server (`~/compose`):
```bash
docker compose pull spectra
docker compose up -d spectra
```

The service is defined in `~/compose/docker-compose.yml` on the host network at
`:38795`. Models are fetched at startup by `ensure_models()` if not already
present in the image.

## Testing an internal endpoint

`:38795` has no public route. Tunnel to it (plain HTTP, no TLS):
```bash
ssh -L 38795:127.0.0.1:38795 kneo@65.108.49.217
curl http://127.0.0.1:38795/analyze -XPOST \
  -H 'Content-Type: application/json' \
  -d '{"soundFragmentId":"<uuid>"}'
```
