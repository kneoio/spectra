"""Transition report for /mix/analyze: what stands out in A's tail and C's head
(per-bar band levels, kicks, vocals, edge shape) and how compatible they are,
for the interactive mixer UI. All times are absolute positions in each file."""

import numpy as np

from mixer import (BAND_GROUPS, BEATS_PER_BAR, Track, _beat_grid, _content_bounds, camelot,
                   keys_compatible, tempo_rate)

EDGE_WINDOW_SEC = 4.0
EDGE_SHAPE_DB = 6.0
VOCAL_CLASH_BARS = 16


def _db(power: float) -> float:
    return round(float(10 * np.log10(power + 1e-12)), 1)


def _overlap(start: float, end: float, segments: list[dict]) -> float:
    """Fraction of [start, end) covered by segments."""
    covered = sum(max(0.0, min(end, s["end"]) - max(start, s["start"])) for s in segments)
    return round(covered / (end - start), 2) if end > start else 0.0


def _bars(track: Track, grid: np.ndarray, vocals: list[dict], offset: float) -> list[dict]:
    """One entry per full bar of the grid inside the segment: mean level overall
    and per band group, plus the share of the bar with vocals."""
    starts = grid[::BEATS_PER_BAR]
    bars = []
    for start, end in zip(starts, starts[1:]):
        frames = (track.times >= start) & (track.times < end)
        if not frames.any():
            continue
        bar = {"start": round(float(start + offset), 3), "end": round(float(end + offset), 3),
               "rms_db": _db(float((10 ** (track.rms_db[frames] / 10)).mean()))}
        for group, names in BAND_GROUPS.items():
            rows = [track.bands.index(n) for n in names]
            bar[f"{group}_db"] = _db(float(track.power[rows][:, frames].sum(axis=0).mean()))
        bar["vocal"] = _overlap(bar["start"], bar["end"], vocals)
        bars.append(bar)
    return bars


def _edge(track: Track, side: str, offset: float) -> dict:
    """A's ending (fade_out / hard_end) or C's opening (build / cold_start), from
    the level change between two EDGE_WINDOW_SEC windows at the edge of the content."""
    start, end = _content_bounds(track)
    w = EDGE_WINDOW_SEC
    if side == "tail":
        change = track.mean_rms_db(end - w, end) - track.mean_rms_db(end - 2 * w, end - w)
        shape = "fade_out" if change <= -EDGE_SHAPE_DB else "hard_end"
    else:
        change = track.mean_rms_db(start + w, start + 2 * w) - track.mean_rms_db(start, start + w)
        shape = "build" if change >= EDGE_SHAPE_DB else "cold_start"
    return {
        "content_start_sec": round(start + offset, 3),
        "content_end_sec": round(end + offset, 3),
        "shape": shape,
        "level_change_db": round(change, 1),
    }


def side_report(m: dict, vocals: list[dict], side: str, grid: str) -> dict:
    """Lanes for one side; `m` is a spectral map, `vocals` its segment-relative
    vocal segments, `side` "tail" (A) or "head" (C)."""
    track = Track(m)
    offset = float(m["segment_offset_sec"])
    grid_times, grid_info = _beat_grid(track, grid)
    vocals_abs = [{"start": round(v["start"] + offset, 3), "end": round(v["end"] + offset, 3)} for v in vocals]
    return {
        "segment": {"start_sec": round(offset, 3),
                    "end_sec": round(offset + float(track.times[-1]) + track.hop, 3)},
        "bpm": track.bpm,
        "key": f"{track.key} {track.scale}",
        "camelot": camelot(track.key, track.scale),
        "grid": grid_info,
        "beats": np.round(grid_times + offset, 3).tolist(),
        "kicks": np.round(track.low_onsets + offset, 3).tolist(),
        "vocals": vocals_abs,
        "bars": _bars(track, grid_times, vocals_abs, offset),
        "edge": _edge(track, side, offset),
    }


def transition_report(map_a: dict, map_c: dict, vocals_a: list[dict], vocals_c: list[dict], grid: str) -> dict:
    a = side_report(map_a, vocals_a, "tail", grid)
    c = side_report(map_c, vocals_c, "head", grid)
    rate, tempo_matched = tempo_rate(a["bpm"], c["bpm"])
    a_last = a["bars"][-VOCAL_CLASH_BARS:]
    c_first = c["bars"][:VOCAL_CLASH_BARS]
    return {
        "a": a,
        "c": c,
        "compatibility": {
            "tempo_diff_pct": round(100 * (c["bpm"] - a["bpm"]) / a["bpm"], 2) if a["bpm"] else None,
            "tempo_matched": tempo_matched,
            "c_playback_rate": round(rate, 4),
            "key_compatible": keys_compatible(a["camelot"], c["camelot"]),
            "vocal_clash_risk": any(b["vocal"] > 0 for b in a_last) and any(b["vocal"] > 0 for b in c_first),
        },
    }
