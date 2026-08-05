#!/usr/bin/env python3
"""Song-essence event timeline (POC) for Mixpla UI animations.

Emits only three event types that capture the feel of a track:

  pulse  — rhythmic heartbeat (kick / beat grid)
  impact — dramatic energy hits (drops, crashes, chorus punches)
  voice  — human / melodic lift (vocal-ish mid-band presence)

Independent of Essentia analysis.

Usage:
    python panns_events.py
    python panns_events.py song.mp3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SAMPLE_RATE = 32_000
CONFIDENCE_THRESHOLD = 0.6
TIMELINE_VERSION = 1
AUDIO_FILE = "/home/aidazi/Music/Clean forever 2022-07-08 1836.wav"

FRAME_LENGTH = 2048
HOP_LENGTH = 512


def load_audio(path: str) -> tuple[np.ndarray, float]:
    """Load mono audio at 32 kHz via ffmpeg."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg not found on PATH; install ffmpeg to decode audio"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode '{path}': {detail}") from exc

    waveform = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if waveform.size == 0:
        raise RuntimeError(f"decoded empty audio from '{path}'")
    return waveform, float(waveform.size / SAMPLE_RATE)


def _band_energies(
    waveform: np.ndarray,
    bands: dict[str, tuple[float, float]],
    sample_rate: int = SAMPLE_RATE,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> dict[str, np.ndarray]:
    """Short-time power in named frequency bands."""
    n = int(waveform.shape[0])
    if n < frame_length:
        return {name: np.zeros(0, dtype=np.float64) for name in bands}

    window = np.hanning(frame_length).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate)
    masks = {
        name: (freqs >= lo) & (freqs <= hi) for name, (lo, hi) in bands.items()
    }
    n_frames = 1 + (n - frame_length) // hop_length
    out = {name: np.empty(n_frames, dtype=np.float64) for name in bands}

    for i in range(n_frames):
        start = i * hop_length
        frame = waveform[start : start + frame_length] * window
        power = np.abs(np.fft.rfft(frame)) ** 2
        for name, mask in masks.items():
            out[name][i] = float(power[mask].sum()) if np.any(mask) else 0.0
    return out


def _smooth(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    return np.convolve(x, kernel, mode="same")


def _positive_log_flux(energy: np.ndarray) -> np.ndarray:
    log_e = np.log(_smooth(energy) + 1e-10)
    return np.maximum(0.0, np.diff(log_e, prepend=log_e[0]))


def _pick_peaks(
    flux: np.ndarray,
    *,
    percentile: float,
    min_interval_sec: float,
    event_type: str,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
) -> list[dict]:
    """Greedy local-max peak picking on an onset-strength curve."""
    if flux.size < 5:
        return []

    thr = float(np.percentile(flux, percentile))
    if thr <= 0:
        return []

    min_gap = max(1, int(round(min_interval_sec * sample_rate / hop_length)))
    peak_ref = float(np.percentile(flux, 99.5)) + 1e-12

    candidates: list[tuple[int, float]] = []
    for i in range(2, flux.size - 2):
        v = float(flux[i])
        if v < thr:
            continue
        if v < flux[i - 1] or v < flux[i + 1]:
            continue
        if v < flux[i - 2] or v < flux[i + 2]:
            continue
        candidates.append((i, v))

    candidates.sort(key=lambda item: item[1], reverse=True)
    chosen: list[tuple[int, float]] = []
    occupied: list[int] = []
    for i, v in candidates:
        if any(abs(i - j) < min_gap for j in occupied):
            continue
        chosen.append((i, v))
        occupied.append(i)
    chosen.sort(key=lambda item: item[0])

    events: list[dict] = []
    for i, v in chosen:
        strength = (v - thr) / (peak_ref - thr + 1e-12)
        confidence = float(np.clip(0.6 + 0.4 * strength, 0.6, 1.0))
        events.append(
            {
                "time": round(i * hop_length / sample_rate, 2),
                "type": event_type,
                "confidence": round(confidence, 2),
            }
        )
    return events


def detect_pulse(band_energy: np.ndarray) -> list[dict]:
    """Rhythmic heartbeat from kick-band onsets (~40–120 Hz)."""
    flux = _positive_log_flux(band_energy)
    return _pick_peaks(
        flux, percentile=92.0, min_interval_sec=0.22, event_type="pulse"
    )


def detect_impact(
    full_energy: np.ndarray,
    low_energy: np.ndarray,
    mid_energy: np.ndarray,
) -> list[dict]:
    """Dramatic hits: strong broadband jumps that aren't kick-only.

    Requires mid-band participation so ordinary kick pulses don't flood this.
    """
    full_flux = _positive_log_flux(full_energy)
    mid_flux = _positive_log_flux(mid_energy)
    low_flux = _positive_log_flux(low_energy)

    # Prefer moments where mid/high also jump, not just the kick band.
    score = full_flux * (0.35 + 0.65 * (mid_flux / (mid_flux.max() + 1e-12)))
    # Down-weight pure low-band spikes.
    low_norm = low_flux / (low_flux.max() + 1e-12)
    mid_norm = mid_flux / (mid_flux.max() + 1e-12)
    score = score * (0.25 + 0.75 * np.clip(mid_norm - 0.35 * low_norm, 0.0, 1.0))

    return _pick_peaks(
        score, percentile=97.5, min_interval_sec=1.8, event_type="impact"
    )


def detect_voice(mid_energy: np.ndarray, full_energy: np.ndarray) -> list[dict]:
    """Vocal / melodic lift: sustained mid-band presence onsets.

    Uses mid-band (300–3400 Hz) relative to full energy, then keeps onsets of
    that ratio — sparse enough to mark section-level 'human/lift' moments.
    """
    mid = _smooth(mid_energy)
    full = _smooth(full_energy) + 1e-10
    ratio = mid / full

    # Onset of relative mid presence (voice-ish lift), not every syllable.
    flux = _positive_log_flux(ratio + 1e-6)

    # Also require absolute mid energy so silence→noise doesn't count.
    mid_gate = mid >= np.percentile(mid, 55)
    gated = flux.copy()
    gated[~mid_gate] = 0.0

    return _pick_peaks(
        gated, percentile=96.0, min_interval_sec=2.5, event_type="voice"
    )


def generate_timeline(audio_path: str) -> dict:
    waveform, duration = load_audio(audio_path)

    bands = _band_energies(
        waveform,
        {
            "low": (40.0, 120.0),
            "mid": (300.0, 3400.0),
            "full": (20.0, 12_000.0),
        },
    )

    events: list[dict] = []
    events.extend(detect_pulse(bands["low"]))
    events.extend(detect_impact(bands["full"], bands["low"], bands["mid"]))
    events.extend(detect_voice(bands["mid"], bands["full"]))
    events.sort(key=lambda e: (e["time"], e["type"]))

    return {
        "version": TIMELINE_VERSION,
        "duration": round(duration, 2),
        "events": events,
    }


def save_timeline(timeline: dict, audio_path: str) -> Path:
    out_path = Path(audio_path).resolve().parent / "timeline.json"
    out_path.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")
    return out_path


def print_events(timeline: dict) -> None:
    events = timeline["events"]
    counts = Counter(e["type"] for e in events)
    print(f"duration: {timeline['duration']:.2f}s")
    print(
        f"events:   {len(events)}  "
        f"(pulse={counts.get('pulse', 0)}, "
        f"impact={counts.get('impact', 0)}, "
        f"voice={counts.get('voice', 0)})"
    )
    print("-" * 40)
    for event in events:
        print(
            f"{event['time']:8.2f}  {event['type']:<8}  {event['confidence']:.2f}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pulse/impact/voice essence timeline for Mixpla."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=AUDIO_FILE,
        help=f"Path to an audio file (default: {AUDIO_FILE})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the event list (still writes timeline.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audio_path = args.audio

    if not Path(audio_path).is_file():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        return 1

    try:
        timeline = generate_timeline(audio_path)
    except Exception as exc:  # noqa: BLE001 - CLI surface
        print(f"Error analyzing '{audio_path}': {exc}", file=sys.stderr)
        return 1

    out_path = save_timeline(timeline, audio_path)
    if not args.quiet:
        print_events(timeline)
        print("-" * 40)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
