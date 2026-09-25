"""Mix plan for A (outgoing, tail map) -> C (incoming, head map) from spectra
spectral maps: beat-aligned entry/exit points, tempo rate, gain match and
per-band-group crossfade automation placed where the two maps clash least."""

import numpy as np

SCORE_BANDS = ("sub", "bass")
BAND_GROUPS = {"low": ("sub", "bass"), "mid": ("low_mid", "mid"), "high": ("high_mid", "presence", "air")}
OVERLAP_BEATS = (8, 16, 32, 64)
C_ENTRY_BEATS = 8
BEATS_PER_BAR = 4
MAX_TEMPO_SHIFT = 0.08
RATE_RETURN_BEATS = 32
MIN_GRID_STRENGTH = 0.3
SILENCE_DB = 30.0
CONTEXT_SEC = 2.0
MAX_GAIN_DB = 6.0
MUTE_DB = -60.0
CUT_WEIGHT = 0.3
ALTERNATIVES = 3


class Track:
    def __init__(self, m: dict):
        self.times = np.asarray(m["times"], dtype=float)
        self.hop = float(m["hop_sec"])
        self.bands = [b["name"] for b in m["bands"]]
        self.power = 10 ** (np.array([m["levels_db"][n] for n in self.bands], dtype=float) / 10)
        self.rms_db = np.asarray(m["rms_db"], dtype=float)
        self.beats = np.asarray(m["beats"], dtype=float)
        self.low_onsets = np.asarray(m.get("low_onsets", []), dtype=float)
        self.bpm = float(m["bpm"])
        self.key = m["key"]
        self.scale = m["scale"]

    def index(self, t):
        i = np.round((np.asarray(t, dtype=float) - self.times[0]) / self.hop).astype(int)
        return np.clip(i, 0, len(self.times) - 1)

    def mean_rms_db(self, t0: float, t1: float) -> float:
        i0, i1 = self.index(t0), self.index(t1)
        p = 10 ** (self.rms_db[i0:max(i1, i0 + 1)] / 10)
        return float(10 * np.log10(p.mean() + 1e-12))

    def group_power(self, group: str, t) -> np.ndarray:
        rows = [self.bands.index(n) for n in BAND_GROUPS[group]]
        return self.power[rows][:, self.index(t)].sum(axis=0)


def tempo_rate(bpm_a: float, bpm_c: float) -> tuple[float, bool]:
    if bpm_a <= 0 or bpm_c <= 0:
        return 1.0, False
    rate = min((bpm_a / bpm_c * f for f in (0.5, 1.0, 2.0)), key=lambda r: abs(r - 1))
    if abs(rate - 1) > MAX_TEMPO_SHIFT:
        return 1.0, False
    return rate, True


PITCH_CLASSES = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6,
                 "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def camelot(key: str, scale: str) -> str:
    pc = PITCH_CLASSES[key]
    if scale == "minor":
        return f"{(7 * (pc + 3) + 7) % 12 + 1}A"
    return f"{(7 * pc + 7) % 12 + 1}B"


def keys_compatible(a: str, c: str) -> bool:
    na, la, nc, lc = int(a[:-1]), a[-1], int(c[:-1]), c[-1]
    if na == nc:
        return True
    return la == lc and (na - nc) % 12 in (1, 11)


def _phase(anchors: np.ndarray, period: float) -> tuple[float, float]:
    """Circular mean of anchor times modulo the beat period: (phase_sec, strength 0..1)."""
    z = np.exp(2j * np.pi * (anchors % period) / period).mean()
    return float((np.angle(z) % (2 * np.pi)) / (2 * np.pi) * period), float(abs(z))


def _beat_grid(track: Track) -> tuple[np.ndarray, dict]:
    """Regular grid at the track's BPM. Phased on low-band onsets (kick hits)
    when they are periodic enough, otherwise on detected beats."""
    period = 60 / track.bpm
    t0, t1 = track.times[0], track.times[-1]
    source, phase, strength = "none", t0, 0.0
    if len(track.low_onsets):
        phase, strength = _phase(track.low_onsets, period)
        source = "low_onsets"
    if strength < MIN_GRID_STRENGTH and len(track.beats):
        phase, strength = _phase(track.beats, period)
        source = "beats"
    first = phase + np.ceil((t0 - phase) / period) * period
    return np.arange(first, t1, period), {"source": source, "strength": round(strength, 3)}


def _content_bounds(track: Track) -> tuple[float, float]:
    loud = np.nonzero(track.rms_db > track.rms_db.max() - SILENCE_DB)[0]
    return float(track.times[loud[0]]), float(track.times[loud[-1]])


def _score(a: Track, c: Track, a_in: float, c_in: float, length: float, rate: float, a_end: float) -> dict:
    k = np.arange(int(length / a.hop)) * a.hop
    rows = [a.bands.index(n) for n in SCORE_BANDS]
    pa = a.power[rows][:, a.index(a_in + k)]
    pc = c.power[rows][:, c.index(c_in + k * rate)]

    a_ctx = a.mean_rms_db(a_in - CONTEXT_SEC, a_in)
    c_out = c_in + length * rate
    c_ctx = c.mean_rms_db(c_out, c_out + CONTEXT_SEC)
    gain_db = float(np.clip(a_ctx - c_ctx, -MAX_GAIN_DB, MAX_GAIN_DB))
    pc = pc * 10 ** (gain_db / 10)

    masking = float(2 * np.minimum(pa, pc).sum() / ((pa + pc).sum() + 1e-12))
    continuity = abs(a_ctx - c_ctx - gain_db) / 12
    cut = CUT_WEIGHT * max(0.0, a_end - (a_in + length)) / max(a_end - a.times[0], 1e-6)
    return {
        "total": float(masking + continuity + cut),
        "masking": masking,
        "continuity": continuity,
        "cut": float(cut),
        "c_gain_db": gain_db,
    }


def _clash_center(a: Track, c: Track, group: str, a_in: float, c_in: float, rate: float,
                  beat: float, n_beats: int) -> float:
    """Bar boundary inside the overlap where the group's combined energy of A and
    C (over one bar around it) is lowest."""
    bars = [j * beat for j in range(BEATS_PER_BAR, n_beats, BEATS_PER_BAR)] or [n_beats * beat / 2]
    window = np.arange(-BEATS_PER_BAR / 2, BEATS_PER_BAR / 2, a.hop / beat) * beat

    def energy(t: float) -> float:
        return float((a.group_power(group, a_in + t + window)
                      + c.group_power(group, c_in + (t + window) * rate)).mean())

    return min(bars, key=energy)


def _keyframe(t: float, a_db: float, c_db: float, a_in: float, c_in: float, rate: float) -> dict:
    return {
        "t": round(float(t), 3),
        "a_time": round(a_in + t, 3),
        "c_time": round(c_in + t * rate, 3),
        "a_db": round(float(max(a_db, MUTE_DB)), 1),
        "c_db": round(float(max(c_db, MUTE_DB)), 1),
    }


def _automation(a: Track, c: Track, a_in: float, c_in: float, rate: float, beat: float, n_beats: int) -> dict:
    length = n_beats * beat
    kf = lambda t, ad, cd: _keyframe(t, ad, cd, a_in, c_in, rate)
    result = {}

    swap = _clash_center(a, c, "low", a_in, c_in, rate, beat, n_beats)
    ramp = beat / 8
    result["low"] = [
        kf(0, 0, MUTE_DB),
        kf(swap - ramp, 0, MUTE_DB),
        kf(swap + ramp, MUTE_DB, 0),
        kf(length, MUTE_DB, 0),
    ]

    for group in ("mid", "high"):
        center = _clash_center(a, c, group, a_in, c_in, rate, beat, n_beats)
        half = min(BEATS_PER_BAR * beat, center, length - center)
        frames = [kf(0, 0, MUTE_DB)]
        for f in np.linspace(0, 1, 5):
            gain_a = 20 * np.log10(max(np.cos(f * np.pi / 2), 1e-6))
            gain_c = 20 * np.log10(max(np.sin(f * np.pi / 2), 1e-6))
            frames.append(kf(center - half + 2 * half * f, gain_a, gain_c))
        frames.append(kf(length, MUTE_DB, 0))
        result[group] = [f for i, f in enumerate(frames) if i == 0 or f["t"] > frames[i - 1]["t"]]

    return result


def _tempo(c: Track, c_in: float, rate: float, length: float) -> list[dict]:
    """C playback rate on C's own timeline: held during the overlap, then ramped
    linearly back to 1.0 over RATE_RETURN_BEATS of C."""
    c_out = c_in + length * rate
    frames = [
        {"c_time": round(c_in, 3), "rate": round(rate, 4)},
        {"c_time": round(c_out, 3), "rate": round(rate, 4)},
    ]
    if rate != 1.0:
        frames.append({"c_time": round(c_out + RATE_RETURN_BEATS * 60 / c.bpm, 3), "rate": 1.0})
    return frames


def plan_mix(map_a: dict, map_c: dict) -> dict:
    a, c = Track(map_a), Track(map_c)
    rate, tempo_matched = tempo_rate(a.bpm, c.bpm)
    beat = 60 / a.bpm
    _, a_end = _content_bounds(a)
    c_start, _ = _content_bounds(c)

    (a_grid, a_grid_info), (c_grid, c_grid_info) = _beat_grid(a), _beat_grid(c)
    a_beats = a_grid[a_grid <= a_end]
    c_entries = c_grid[c_grid >= c_start - c.hop][:C_ENTRY_BEATS]
    if not len(c_entries):
        c_entries = np.array([c_start])

    candidates = []
    for n_beats in OVERLAP_BEATS:
        length = n_beats * beat
        for c_in in c_entries:
            if c_in + length * rate > c.times[-1]:
                continue
            for a_in in a_beats:
                if a_in + length > a_end + a.hop:
                    continue
                score = _score(a, c, float(a_in), float(c_in), length, rate, a_end)
                candidates.append((score, float(a_in), float(c_in), n_beats))
    if not candidates:
        raise ValueError("segments too short for any beat-aligned overlap")

    candidates.sort(key=lambda x: x[0]["total"])
    score, a_in, c_in, n_beats = candidates[0]
    length = n_beats * beat
    cam_a, cam_c = camelot(a.key, a.scale), camelot(c.key, c.scale)

    return {
        "a": {
            "mix_start_sec": round(a_in, 3),
            "stop_sec": round(a_in + length, 3),
            "bpm": a.bpm,
            "key": f"{a.key} {a.scale}",
            "camelot": cam_a,
            "grid": a_grid_info,
        },
        "c": {
            "start_from_sec": round(c_in, 3),
            "playback_rate": round(rate, 4),
            "gain_db": round(score["c_gain_db"], 1) + 0.0,
            "bpm": c.bpm,
            "key": f"{c.key} {c.scale}",
            "camelot": cam_c,
            "grid": c_grid_info,
        },
        "overlap_beats": n_beats,
        "overlap_sec": round(length, 3),
        "tempo_matched": tempo_matched,
        "key_compatible": keys_compatible(cam_a, cam_c),
        "score": {k: round(v, 4) for k, v in score.items() if k != "c_gain_db"},
        "tempo": _tempo(c, c_in, rate, length),
        "automation": _automation(a, c, a_in, c_in, rate, beat, n_beats),
        "alternatives": [
            {
                "a_mix_start_sec": round(ai, 3),
                "c_start_from_sec": round(ci, 3),
                "overlap_beats": nb,
                "score": round(s["total"], 4),
            }
            for s, ai, ci, nb in candidates[1:1 + ALTERNATIVES]
        ],
    }
