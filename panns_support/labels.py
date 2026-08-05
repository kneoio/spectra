"""AudioSet class labels shipped with the POC (no network required)."""

from __future__ import annotations

import csv
from pathlib import Path

_LABELS_CSV = Path(__file__).resolve().parent / "class_labels_indices.csv"


def load_labels(csv_path: Path | None = None) -> list[str]:
    path = csv_path or _LABELS_CSV
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    # index,mid,display_name
    return [row[2] for row in rows[1:]]


labels = load_labels()
classes_num = len(labels)
