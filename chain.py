"""Chain mix A -> B1..Bn -> C where the B tracks are short effects/voice that
are not analyzed. A and the B tracks are stitched back to back (or crossfaded
by head_sec when > 0); only the last B is mixed into C, by an equal-power
crossfade over a window of min(tail_sec, BRIDGE_FRACTION * its duration), so a
short B keeps its recognizable opening before it dissolves into C."""

from typing import Callable

import numpy as np
import soundfile as sf

from render import BANDS, MUTE_DB, RENDER_SR, load_audio, mix_audio, write_audio

TAIL_SEC = 5.0
HEAD_SEC = 0.0
BRIDGE_FRACTION = 0.5
BRIDGE_STEPS = 5


def bridge_seconds(duration: float, tail_sec: float = TAIL_SEC) -> float:
    return min(tail_sec, BRIDGE_FRACTION * duration)


def plan_bridge(duration: float, tail_sec: float = TAIL_SEC) -> dict:
    """render.mix_audio-compatible plan for the last B (of `duration` sec) into C:
    B stops at its end, C starts at its beginning, both at rate 1.0."""
    if duration <= 0 or tail_sec <= 0:
        raise ValueError("duration and tail_sec must be positive")
    window = bridge_seconds(duration, tail_sec)
    start = duration - window

    frames = []
    for f in np.linspace(0, 1, BRIDGE_STEPS):
        gain_b = 20 * np.log10(max(np.cos(f * np.pi / 2), 1e-6))
        gain_c = 20 * np.log10(max(np.sin(f * np.pi / 2), 1e-6))
        frames.append({
            "t": round(float(f * window), 3),
            "a_time": round(start + float(f * window), 3),
            "c_time": round(float(f * window), 3),
            "a_db": round(float(max(gain_b, MUTE_DB)), 1),
            "c_db": round(float(max(gain_c, MUTE_DB)), 1),
        })
    return {
        "a": {"mix_start_sec": round(start, 3), "stop_sec": round(duration, 3)},
        "c": {"start_from_sec": 0.0, "playback_rate": 1.0, "gain_db": 0.0},
        "overlap_sec": round(window, 3),
        "tempo": [{"c_time": 0.0, "rate": 1.0}, {"c_time": round(window, 3), "rate": 1.0}],
        "automation": {band: frames for band in BANDS},
    }


def _join(first: np.ndarray, second: np.ndarray, overlap_sec: float) -> np.ndarray:
    """Stitch `second` after `first`; overlap_sec > 0 equal-power crossfades the
    seam, 0 is a hard cut."""
    n = min(int(round(overlap_sec * RENDER_SR)), len(first), len(second))
    if n <= 0:
        return np.concatenate([first, second], axis=0)
    fade = np.linspace(0, np.pi / 2, n)[:, None]
    seam = first[-n:] * np.cos(fade) + second[:n] * np.sin(fade)
    return np.concatenate([first[:-n], seam, second[n:]], axis=0)


def render_chain(path_a: str, paths_b: list[str], path_c: str, out_path: str,
                 head_sec: float = HEAD_SEC, tail_sec: float = TAIL_SEC,
                 progress: Callable[[str], None] | None = None, fmt: str = "opus") -> dict:
    """Render A, the B tracks in order, then C to `out_path`; returns the bridge plan.
    Rate is always 1.0, so no time-stretching (and no rubberband) is involved.
    `progress` is called with a short message before each stage."""
    say = progress or (lambda _: None)
    if not paths_b:
        raise ValueError("at least one B track is required")
    if head_sec < 0:
        raise ValueError("head_sec must be >= 0")

    say(f"Stitching A and {len(paths_b)} B track(s)")
    audio = load_audio(path_a)
    for path_b in paths_b[:-1]:
        audio = _join(audio, load_audio(path_b), head_sec)

    last_b = paths_b[-1]
    plan = plan_bridge(sf.info(last_b).duration, tail_sec)
    say(f"Mixing the last B into C over {plan['overlap_sec']:g}s")
    audio = _join(audio, mix_audio(plan, last_b, path_c, keylock=False), head_sec)

    say("Writing the mix")
    write_audio(audio, out_path, fmt)
    return plan
