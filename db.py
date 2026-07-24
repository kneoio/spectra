import json
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

# datanest's DB (moon). Spectra only UPDATEs the analysis result onto an
# existing SoundFragment row — it never inserts or deletes.
DB_HOST = os.environ.get("SPECTRA_DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("SPECTRA_DB_PORT", "8572"))
DB_NAME = os.environ.get("SPECTRA_DB_NAME", "moon")
DB_USER = os.environ.get("SPECTRA_DB_USER", "regolith")
DB_PASSWORD = os.environ.get("SPECTRA_DB_PASSWORD")

SF_TABLE = "mixpla__sound_fragments"


def save_analysis(sound_fragment_id: str, metadata: dict) -> int:
    """Write the analysis result into the SoundFragment's `add_info` jsonb column.
    Returns the number of rows updated (0 if the id doesn't exist)."""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                f"UPDATE {SF_TABLE} SET add_info = %s WHERE id = %s",
                (json.dumps(metadata), sound_fragment_id),
            )
            return cur.rowcount
    finally:
        conn.close()
