"""Transition report for /mix/analyze: what stands out in A's tail and C's head
(per-bar band levels, kicks, vocals, edge shape) and how compatible they are,
for the interactive mixer UI. All times are absolute positions in each file.

Run as a script to cut those pieces out of two full songs and POST only the
pieces to /mix:

    python transition.py song_a song_c
"""

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


def _post_pieces(url: str, a_path: str, c_path: str, seconds: float, grid: str,
                 fmt: str, timeout: float) -> tuple[bytes, dict]:
    """POST the already-cut tail and head. The mix that comes back is only those pieces."""
    import json
    import urllib.error
    import urllib.request

    from mix_client import multipart

    body, content_type = multipart(
        {"seconds": f"{seconds:.3f}", "grid": grid, "format": fmt},
        {"file_a": a_path, "file_c": c_path},
    )
    req = urllib.request.Request(f"{url.rstrip('/')}/mix", data=body, method="POST",
                                 headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            plan_header = resp.headers.get("X-Mix-Plan")
            audio = resp.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"server returned {e.code}: {e.read().decode(errors='replace')}") from e
    if not plan_header:
        raise SystemExit("response has no X-Mix-Plan header")
    return audio, json.loads(plan_header)


def main() -> None:
    import argparse
    import json
    import os
    import tempfile

    from mix_client import DEFAULT_EXCERPT_SEC, DEFAULT_URL, cut, duration_sec

    parser = argparse.ArgumentParser(
        description="Cut A's tail and C's head from two full songs and POST only those pieces to Spectra /mix.")
    parser.add_argument("song_a", help="outgoing song (full file)")
    parser.add_argument("song_c", help="incoming song (full file)")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="mix_out", help="output directory")
    parser.add_argument("--excerpt", type=float, default=DEFAULT_EXCERPT_SEC,
                        help="seconds cut from A's tail and C's head, and the analysis window")
    parser.add_argument("--grid", default="beats", choices=["beats", "kicks"])
    parser.add_argument("--format", default="wav", choices=["opus", "wav"])
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    a_offset = max(0.0, duration_sec(args.song_a) - args.excerpt)
    with tempfile.TemporaryDirectory() as tmp:
        excerpt_a = os.path.join(tmp, "a_tail.ogg")
        excerpt_c = os.path.join(tmp, "c_head.ogg")
        cut(args.song_a, a_offset, args.excerpt, excerpt_a)
        cut(args.song_c, 0.0, args.excerpt, excerpt_c)
        sent = os.path.getsize(excerpt_a) + os.path.getsize(excerpt_c)
        full = os.path.getsize(args.song_a) + os.path.getsize(args.song_c)
        print(f"sending {sent / 1e6:.2f} MB of pieces (full songs: {full / 1e6:.2f} MB), A tail from {a_offset:.1f}s")
        audio, plan = _post_pieces(args.url, excerpt_a, excerpt_c, args.excerpt, args.grid, args.format, args.timeout)

    mix_path = os.path.join(args.out, f"mix.{args.format}")
    with open(mix_path, "wb") as f:
        f.write(audio)
    with open(os.path.join(args.out, "plan.json"), "w") as f:
        json.dump(plan, f, indent=2)
    print(f"mix: {mix_path} ({len(audio) / 1e6:.2f} MB)")
    print(json.dumps({"a": plan["a"], "c": plan["c"], "overlap_sec": plan.get("overlap_sec")}, indent=2))


if __name__ == "__main__":
    import subprocess
    import sys

    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"command failed: {' '.join(map(str, e.cmd))}")
