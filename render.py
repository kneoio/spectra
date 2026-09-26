"""Render a coincidense mix plan (see mixer.plan_mix) to a WAV file: crossfade
A into C on three bands per the plan's automation, warping C's speed to the
plan's tempo curve (time-stretched with rubberband, or resampled vinyl-style)."""

import argparse
import json
import math
import shutil
import sys

import numpy as np
import pyrubberband as pyrb
import soundfile as sf
from scipy import signal

RENDER_SR = 44100
OPUS_SR = 48000
OUTPUT_FORMATS = ("opus", "wav")
LOW_CUTOFF_HZ = 250.0
HIGH_CUTOFF_HZ = 2000.0
RECON_ERROR_DB_MAX = -60.0
MUTE_DB = -60.0
PEAK_TARGET_DBFS = -0.3
RATE_EPSILON = 1e-3
RAMP_SUBSEGMENTS = 8
BANDS = ("low", "mid", "high")


def check_rubberband() -> None:
    """Raise if the `rubberband` CLI (required by pyrubberband) isn't on PATH."""
    if shutil.which("rubberband") is None:
        raise RuntimeError(
            "rubberband CLI not found on PATH. Install it (e.g. "
            "`sudo apt install rubberband-cli`) or render with keylock=False."
        )


def _to_stereo(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x[:, :2]


def _resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return x
    g = math.gcd(sr, target_sr)
    return signal.resample_poly(x, target_sr // g, sr // g, axis=0)


def load_audio(path: str, target_sr: int = RENDER_SR) -> np.ndarray:
    """Stereo float64 audio at `target_sr`, mono upmixed and resampled as needed."""
    data, sr = sf.read(path, dtype="float64", always_2d=True)
    return _resample(_to_stereo(data), sr, target_sr)


def _pad_or_trim(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    return np.concatenate([x, np.zeros((n - len(x), x.shape[1]), dtype=x.dtype)], axis=0)


def _lr_lowpass(x: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
    """4th-order Linkwitz-Riley lowpass: two cascaded 2nd-order Butterworths, zero-phase."""
    sos = signal.butter(2, cutoff, btype="low", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, signal.sosfiltfilt(sos, x, axis=0), axis=0)


def split_bands(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Low (<250Hz) / mid (250-2000Hz) / high (>2000Hz), summing back to `x`."""
    low = _lr_lowpass(x, LOW_CUTOFF_HZ, sr)
    above_low = x - low
    mid = _lr_lowpass(above_low, HIGH_CUTOFF_HZ, sr)
    high = above_low - mid
    return low, mid, high


def check_reconstruction(x: np.ndarray, low: np.ndarray, mid: np.ndarray, high: np.ndarray) -> None:
    recon = low + mid + high
    ref = float(np.sqrt(np.mean(x ** 2))) + 1e-12
    err = float(np.sqrt(np.mean((x - recon) ** 2))) + 1e-12
    db = 20 * np.log10(err / ref)
    if db >= RECON_ERROR_DB_MAX:
        raise AssertionError(f"band reconstruction error {db:.1f} dB exceeds {RECON_ERROR_DB_MAX} dB")


def _db_to_gain(db: np.ndarray) -> np.ndarray:
    return np.where(db <= MUTE_DB, 0.0, 10 ** (db / 20))


def _band_gain(points: list[dict], db_key: str, sr: int, n_samples: int, t0: float, left: float | None = None) -> np.ndarray:
    times = np.array([p["t"] for p in points], dtype=float)
    dbs = np.array([p[db_key] for p in points], dtype=float)
    sample_t = t0 + np.arange(n_samples) / sr
    kwargs = {} if left is None else {"left": left}
    return _db_to_gain(np.interp(sample_t, times, dbs, **kwargs))


def _shaped(x: np.ndarray, automation: dict, db_key: str, sr: int, t0: float, left: float | None = None) -> np.ndarray:
    low, mid, high = split_bands(x, sr)
    check_reconstruction(x, low, mid, high)
    bands = {"low": low, "mid": mid, "high": high}
    out = np.zeros_like(x)
    for band in BANDS:
        gain = _band_gain(automation[band], db_key, sr, len(x), t0, left)
        out += bands[band] * gain[:, None]
    return out


def _rate_segments(tempo: list[dict], c_duration: float) -> list[tuple[float, float, float, float]]:
    """(c_start, c_end, rate_start, rate_end) pieces from `tempo`, extended with
    a final rate=1.0 piece from the last tempo point to the end of the source."""
    pts = sorted(tempo, key=lambda p: p["c_time"])
    segments = [(p0["c_time"], p1["c_time"], p0["rate"], p1["rate"]) for p0, p1 in zip(pts, pts[1:])]
    last_t = pts[-1]["c_time"]
    if last_t < c_duration:
        segments.append((last_t, c_duration, 1.0, 1.0))
    return segments


def _expand_ramp_segments(segments: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float]]:
    """Split into (c_start, c_end, rate) pieces, subdividing non-constant (ramp)
    segments into near-constant sub-pieces so each can be time-stretched at a
    single ratio."""
    expanded = []
    for c0, c1, r0, r1 in segments:
        if abs(r1 - r0) < RATE_EPSILON or c1 <= c0:
            expanded.append((c0, c1, (r0 + r1) / 2))
            continue
        edges = np.linspace(c0, c1, RAMP_SUBSEGMENTS + 1)
        for a0, a1 in zip(edges, edges[1:]):
            ra = r0 + (r1 - r0) * (a0 - c0) / (c1 - c0)
            rb = r0 + (r1 - r0) * (a1 - c0) / (c1 - c0)
            expanded.append((float(a0), float(a1), float((ra + rb) / 2)))
    return expanded


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    if len(x) < 2:
        return _pad_or_trim(x, n)
    idx = np.linspace(0, len(x) - 1, n)
    src = np.arange(len(x))
    return np.stack([np.interp(idx, src, x[:, ch]) for ch in range(x.shape[1])], axis=1)


def _warp_resample(c_audio: np.ndarray, sr: int, segments: list[tuple[float, float, float]]) -> np.ndarray:
    src = np.arange(len(c_audio))
    chunks = []
    for c0, c1, rate in segments:
        n = int(round((c1 - c0) / rate * sr))
        if n <= 0:
            continue
        pos = c0 * sr + rate * np.arange(n)
        chunks.append(np.stack([np.interp(pos, src, c_audio[:, ch]) for ch in range(c_audio.shape[1])], axis=1))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, c_audio.shape[1]))


def _warp_keylock(c_audio: np.ndarray, sr: int, segments: list[tuple[float, float, float]]) -> np.ndarray:
    chunks = []
    for c0, c1, rate in segments:
        s0, s1 = int(round(c0 * sr)), min(int(round(c1 * sr)), len(c_audio))
        if s1 <= s0:
            continue
        seg = c_audio[s0:s1]
        target_n = max(int(round((c1 - c0) / rate * sr)), 1)
        stretched = seg if abs(rate - 1.0) < RATE_EPSILON else pyrb.time_stretch(seg, sr, rate)
        chunks.append(_fit_length(stretched, target_n))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, c_audio.shape[1]))


def build_c_output(c_audio: np.ndarray, sr: int, plan: dict, keylock: bool) -> np.ndarray:
    """C's audio from `start_from_sec` to end of file, warped to follow the
    plan's tempo curve. If the plan holds rate 1.0 throughout, no processing."""
    tempo = plan["tempo"]
    start_sample = int(round(plan["c"]["start_from_sec"] * sr))
    c_from_start = c_audio[start_sample:]
    if all(abs(p["rate"] - 1.0) < RATE_EPSILON for p in tempo):
        return c_from_start.copy()

    duration = len(c_from_start) / sr
    local_tempo = [{"c_time": p["c_time"] - plan["c"]["start_from_sec"], "rate": p["rate"]} for p in tempo]
    segments = _expand_ramp_segments(_rate_segments(local_tempo, duration))
    warp = _warp_keylock if keylock else _warp_resample
    return warp(c_from_start, sr, segments)


def mix_audio(plan: dict, path_a: str, path_c: str, keylock: bool = True) -> np.ndarray:
    """The mixed A->C audio (stereo, RENDER_SR), before peak normalization."""
    if keylock:
        check_rubberband()

    sr = RENDER_SR
    automation = plan["automation"]

    a_audio = _pad_or_trim(load_audio(path_a, sr), int(round(plan["a"]["stop_sec"] * sr)))
    a_mixed = _shaped(a_audio, automation, "a_db", sr, t0=-plan["a"]["mix_start_sec"], left=0.0)

    c_audio = load_audio(path_c, sr)
    c_out = build_c_output(c_audio, sr, plan, keylock)
    c_mixed = _shaped(c_out, automation, "c_db", sr, t0=0.0)

    c_start_sample = int(round(plan["a"]["mix_start_sec"] * sr))
    total_len = max(len(a_mixed), c_start_sample + len(c_mixed))
    mix = np.zeros((total_len, 2), dtype=np.float64)
    mix[:len(a_mixed)] += a_mixed
    mix[c_start_sample:c_start_sample + len(c_mixed)] += c_mixed
    return mix


def write_audio(mix: np.ndarray, out_path: str, fmt: str = "wav", sr: int = RENDER_SR) -> None:
    """Peak-limit `mix` to PEAK_TARGET_DBFS and write it as 24-bit WAV or Ogg
    Opus (resampled to OPUS_SR, which Opus requires)."""
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"unsupported format {fmt!r}; use one of {', '.join(OUTPUT_FORMATS)}")
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    target_peak = 10 ** (PEAK_TARGET_DBFS / 20)
    if peak > target_peak:
        mix = mix * (target_peak / peak)
    if fmt == "opus":
        sf.write(out_path, _resample(mix, sr, OPUS_SR), OPUS_SR, format="OGG", subtype="OPUS")
    else:
        sf.write(out_path, mix, sr, subtype="PCM_24")


def render(plan: dict, path_a: str, path_c: str, out_path: str, keylock: bool = True, fmt: str = "wav") -> None:
    write_audio(mix_audio(plan, path_a, path_c, keylock), out_path, fmt)


def main() -> None:
    parser = argparse.ArgumentParser(prog="coincidense.render")
    parser.add_argument("plan", help="mix plan JSON file, as produced by coincidense --out")
    parser.add_argument("a", help="outgoing track (same file used to build the plan)")
    parser.add_argument("c", help="incoming track (same file used to build the plan)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-keylock", action="store_true")
    args = parser.parse_args()

    with open(args.plan) as f:
        plan = json.load(f)

    try:
        render(plan, args.a, args.c, args.out, keylock=not args.no_keylock)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
