import json
import sys

from analysis import analyze
from models_setup import ensure_models

AUDIO_FILE = "/home/aidazi/Music/La Monstro Konas.wav"


def main():
    ensure_models()

    try:
        result = analyze(AUDIO_FILE)
    except RuntimeError as e:
        print(f"Error analyzing '{AUDIO_FILE}': {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
