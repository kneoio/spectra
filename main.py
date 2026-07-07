import json
import os
import sys

AUDIO_FILE = "/home/aidazi/Music/La Monstro Konas.wav"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import essentia
import essentia.standard as es
import numpy as np

essentia.log.infoActive = False
essentia.log.warningActive = False

from check_metadata import scan_file as scan_metadata_for_ai_tags

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

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


def classify_genre_mood(path: str) -> dict:
    # Discogs-EffNet embedding model expects mono 16kHz audio.
    audio_16k = es.MonoLoader(filename=path, sampleRate=16000)()

    embedding_model = es.TensorflowPredictEffnetDiscogs(
        graphFilename=EMBEDDING_MODEL, output="PartitionedCall:1"
    )
    embeddings = embedding_model(audio_16k)

    genre_labels = load_labels("genre_discogs400-discogs-effnet-1")
    genre_model = es.TensorflowPredict2D(
        graphFilename=GENRE_MODEL,
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0",
    )
    genre_scores = np.mean(genre_model(embeddings), axis=0)
    top_idx = np.argsort(genre_scores)[::-1][:5]
    top_genres = [
        {"genre": genre_labels[i], "score": round(float(genre_scores[i]), 3)}
        for i in top_idx
    ]

    moods = {}
    for mood_name, model_name in MOOD_MODELS.items():
        labels = load_labels(model_name)
        model = es.TensorflowPredict2D(
            graphFilename=os.path.join(MODELS_DIR, f"{model_name}.pb"),
            output="model/Softmax",
        )
        scores = np.mean(model(embeddings), axis=0)
        moods[mood_name] = round(float(scores[labels.index(mood_name)]), 3)

    dance_labels = load_labels(DANCEABILITY_MODEL)
    dance_model = es.TensorflowPredict2D(
        graphFilename=os.path.join(MODELS_DIR, f"{DANCEABILITY_MODEL}.pb"),
        output="model/Softmax",
    )
    dance_scores = np.mean(dance_model(embeddings), axis=0)
    danceability = round(float(dance_scores[dance_labels.index("danceable")]), 3)

    return {
        "top_genres": top_genres,
        "moods": moods,
        "danceability": danceability,
    }


def analyze(path: str) -> dict:
    loader = es.MonoLoader(filename=path)
    audio = loader()

    duration = len(audio) / loader.paramValue("sampleRate")

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
    result.update(classify_genre_mood(path))

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


def main():
    try:
        result = analyze(AUDIO_FILE)
    except RuntimeError as e:
        print(f"Error analyzing '{AUDIO_FILE}': {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
