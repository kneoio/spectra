import json
import os
import subprocess
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import essentia
import essentia.standard as es
import numpy as np

essentia.log.infoActive = False
essentia.log.warningActive = False

from check_metadata import scan_file as scan_metadata_for_ai_tags

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")


def _transcode_to_wav(path: str) -> str:
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(
        [FFMPEG_PATH, "-y", "-i", path, wav_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return wav_path


def _load_mono(path: str, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Load mono audio via a normalized WAV, always transcoded through system
    ffmpeg first. Essentia's bundled decoder can't handle some codecs (e.g.
    Opus) — and a MonoLoader that fails to configure corrupts memory in its
    own cleanup path rather than just raising, so it must never even be
    constructed against a file it might not decode."""
    wav_path = _transcode_to_wav(path)
    try:
        kwargs = {"sampleRate": sample_rate} if sample_rate else {}
        loader = es.MonoLoader(filename=wav_path, **kwargs)
        return loader(), int(loader.paramValue("sampleRate"))
    finally:
        os.remove(wav_path)

EMBEDDING_MODEL = os.path.join(MODELS_DIR, "discogs-effnet-bs64-1.pb")
GENRE_MODEL = os.path.join(MODELS_DIR, "genre_discogs400-discogs-effnet-1.pb")
GENRE_LABELS_FILE = os.path.join(MODELS_DIR, "genre_discogs400-discogs-effnet-1.json")

# mood name -> (model file, label for the "positive" class in that head's json)
MOOD_MODELS = {
    "happy": "mood_happy-discogs-effnet-1",
    "sad": "mood_sad-discogs-effnet-1",
    "relaxed": "mood_relaxed-discogs-effnet-1",
    "aggressive": "mood_aggressive-discogs-effnet-1",
    "party": "mood_party-discogs-effnet-1",
}
DANCEABILITY_MODEL = "danceability-discogs-effnet-1"


def load_labels(model_name: str) -> list[str]:
    with open(os.path.join(MODELS_DIR, f"{model_name}.json")) as f:
        return json.load(f)["classes"]


class ModelBundle:
    """All TF graphs + label sets, built once and reused across analyze() calls."""

    def __init__(self):
        self.embedding_model = es.TensorflowPredictEffnetDiscogs(
            graphFilename=EMBEDDING_MODEL, output="PartitionedCall:1"
        )

        self.genre_labels = load_labels("genre_discogs400-discogs-effnet-1")
        self.genre_model = es.TensorflowPredict2D(
            graphFilename=GENRE_MODEL,
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )

        self.mood_labels = {}
        self.mood_models = {}
        for mood_name, model_name in MOOD_MODELS.items():
            self.mood_labels[mood_name] = load_labels(model_name)
            self.mood_models[mood_name] = es.TensorflowPredict2D(
                graphFilename=os.path.join(MODELS_DIR, f"{model_name}.pb"),
                output="model/Softmax",
            )

        self.dance_labels = load_labels(DANCEABILITY_MODEL)
        self.dance_model = es.TensorflowPredict2D(
            graphFilename=os.path.join(MODELS_DIR, f"{DANCEABILITY_MODEL}.pb"),
            output="model/Softmax",
        )


_bundle: ModelBundle | None = None


def get_model_bundle() -> ModelBundle:
    global _bundle
    if _bundle is None:
        _bundle = ModelBundle()
    return _bundle


def classify_genre_mood(path: str, models: ModelBundle) -> dict:
    # Discogs-EffNet embedding model expects mono 16kHz audio.
    audio_16k, _ = _load_mono(path, sample_rate=16000)
    embeddings = models.embedding_model(audio_16k)

    genre_scores = np.mean(models.genre_model(embeddings), axis=0)
    top_idx = np.argsort(genre_scores)[::-1][:5]
    top_genres = [
        {"genre": models.genre_labels[i], "score": round(float(genre_scores[i]), 3)}
        for i in top_idx
    ]

    moods = {}
    for mood_name, model in models.mood_models.items():
        labels = models.mood_labels[mood_name]
        scores = np.mean(model(embeddings), axis=0)
        moods[mood_name] = round(float(scores[labels.index(mood_name)]), 3)

    dance_scores = np.mean(models.dance_model(embeddings), axis=0)
    danceability = round(float(dance_scores[models.dance_labels.index("danceable")]), 3)

    return {
        "top_genres": top_genres,
        "moods": moods,
        "danceability": danceability,
    }


def analyze(path: str) -> dict:
    models = get_model_bundle()
    audio, sample_rate = _load_mono(path)

    duration = len(audio) / sample_rate

    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

    key_extractor = es.KeyExtractor()
    key, scale, key_strength = key_extractor(audio)

    loudness = es.Loudness()(audio)

    result = {
        "file": path,
        "duration_sec": round(duration, 2),
        "bpm": round(float(bpm), 2),
        "beats_confidence": round(float(beats_confidence), 3),
        "key": key,
        "scale": scale,
        "key_strength": round(float(key_strength), 3),
        "loudness": round(float(loudness), 2),
    }
    result.update(classify_genre_mood(path, models))

    # Metadata-only check (e.g. Suno embeds a "made with suno" comment tag).
    # NOTE: this is a weak signal — tags are trivially stripped or forged by
    # re-encoding, so a "clean" result does NOT mean the track is human-made,
    # it only means no AI-generation tag was found in this file's metadata.
    metadata_scan = scan_metadata_for_ai_tags(path)
    result["ai_generated_metadata_check"] = {
        "suspected_ai_generated": bool(metadata_scan["suspect_hits"]),
        "evidence": metadata_scan["suspect_hits"],
        "caveat": "metadata-only signal; absence does not prove human authorship",
    }

    return result


SPECTRAL_MAP_SR = 44100
SPECTRAL_MAP_FRAME_SIZE = 2048
SPECTRAL_MAP_HOP_SIZE = 1024
LOW_ONSET_CUTOFF_HZ = 250.0

# (low_hz, high_hz) per band, covering 0..sr/2.
BAND_RANGES = {
    "sub": (20, 60),
    "bass": (60, 150),
    "low_mid": (150, 400),
    "mid": (400, 2000),
    "high_mid": (2000, 4000),
    "presence": (4000, 6000),
    "air": (6000, SPECTRAL_MAP_SR / 2),
}


def _segment_audio(audio: np.ndarray, segment: str, seconds: float, sr: int) -> np.ndarray:
    n = len(audio)
    seg_len = min(int(seconds * sr), n)
    if segment == "tail":
        return audio[n - seg_len:]
    if segment == "head":
        return audio[:seg_len]
    raise ValueError(f"segment must be 'tail' or 'head', got {segment!r}")


def spectral_map(path: str, segment: str, seconds: float) -> dict:
    """Band-energy + rhythm time series for the given edge (head/tail) of a
    track, in the schema mixer.Track expects: per-frame band levels_db and
    rms_db, detected beats, low-band onsets, plus bpm/key/scale. All times
    (times/beats/low_onsets) are relative to the extracted segment, starting
    at 0 — segment_offset_sec is where that segment sits in the full file
    (0 for 'head'; full_duration - seconds for 'tail'), needed to convert a
    mix plan's times back to absolute positions in the original file."""
    sr = SPECTRAL_MAP_SR
    audio, _ = _load_mono(path, sample_rate=sr)
    seg = _segment_audio(audio, segment, seconds, sr)
    if len(seg) < SPECTRAL_MAP_FRAME_SIZE:
        raise ValueError(f"segment too short for analysis: {len(seg)} samples at sr={sr}")
    segment_offset_sec = (len(audio) - len(seg)) / sr if segment == "tail" else 0.0

    hop_sec = SPECTRAL_MAP_HOP_SIZE / sr
    windowing = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    rms = es.RMS()

    n_bins = SPECTRAL_MAP_FRAME_SIZE // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    band_bins = {name: np.where((freqs >= lo) & (freqs < hi))[0] for name, (lo, hi) in BAND_RANGES.items()}

    times, rms_db = [], []
    levels_db = {name: [] for name in BAND_RANGES}
    for i, frame in enumerate(es.FrameGenerator(seg, frameSize=SPECTRAL_MAP_FRAME_SIZE,
                                                 hopSize=SPECTRAL_MAP_HOP_SIZE, startFromZero=True)):
        spec = spectrum(windowing(frame))
        times.append(i * hop_sec)
        rms_db.append(20 * np.log10(rms(frame) + 1e-12))
        for name, bins in band_bins.items():
            energy = float(np.sum(spec[bins] ** 2)) if len(bins) else 0.0
            levels_db[name].append(10 * np.log10(energy + 1e-12))

    bpm, beats, beats_confidence, _, _ = es.RhythmExtractor2013(method="multifeature")(seg)

    lowpass = es.LowPass(cutoffFrequency=LOW_ONSET_CUTOFF_HZ, sampleRate=sr)
    low_onsets, _ = es.OnsetRate()(lowpass(seg))

    key, scale, key_strength = es.KeyExtractor()(seg)

    return {
        "segment": segment,
        "segment_offset_sec": round(segment_offset_sec, 4),
        "hop_sec": round(hop_sec, 6),
        "times": [round(t, 4) for t in times],
        "bands": [{"name": name} for name in BAND_RANGES],
        "levels_db": {name: [round(v, 2) for v in vals] for name, vals in levels_db.items()},
        "rms_db": [round(v, 2) for v in rms_db],
        "beats": [round(float(b), 4) for b in beats],
        "beats_confidence": round(float(beats_confidence), 3),
        "low_onsets": [round(float(o), 4) for o in low_onsets],
        "bpm": round(float(bpm), 2),
        "key": key,
        "scale": scale,
        "key_strength": round(float(key_strength), 3),
    }


def is_music(result: dict) -> bool:
    """Heuristic music-vs-speech verdict for pre-save gating.

    Uses Discogs top genre: labels under Non-Music---* (Spoken Word, Dialogue, …)
    mean the file is not a song. Tiny clips (<3s) are rejected as incomplete.
    """
    duration = float(result.get("duration_sec") or 0)
    if duration < 3.0:
        return False
    top = result.get("top_genres") or []
    if not top:
        return True
    genre = (top[0].get("genre") or "") if isinstance(top[0], dict) else ""
    return not str(genre).startswith("Non-Music---")
