import os
import subprocess

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODEL_BASE_URLS = {
    "discogs-effnet-bs64-1.pb": "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
    "genre_discogs400-discogs-effnet-1.pb": "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
    "mood_happy-discogs-effnet-1.pb": "https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb",
    "mood_sad-discogs-effnet-1.pb": "https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-discogs-effnet-1.pb",
    "mood_relaxed-discogs-effnet-1.pb": "https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.pb",
    "mood_aggressive-discogs-effnet-1.pb": "https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.pb",
    "mood_party-discogs-effnet-1.pb": "https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-discogs-effnet-1.pb",
    "danceability-discogs-effnet-1.pb": "https://essentia.upf.edu/models/classification-heads/danceability/danceability-discogs-effnet-1.pb",
}


def ensure_models(quiet: bool = False) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in MODEL_BASE_URLS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            continue
        if not quiet:
            print(f"Downloading model {filename} ...", flush=True)
        tmp_dest = dest + ".part"
        subprocess.run(
            ["curl", "-fsSL", "--max-time", "60", "-o", tmp_dest, url],
            check=True,
        )
        os.rename(tmp_dest, dest)


if __name__ == "__main__":
    ensure_models()
