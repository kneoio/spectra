#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

uv sync
uv run python models_setup.py
uv run uvicorn service:app --host 0.0.0.0 --port 38795
