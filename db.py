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

# _files.file_type: 101 = original SOUND_FRAGMENT, 102 = OPUS_ENCODED, 0 = legacy
# original (pre-typing). We must analyze the ORIGINAL, never the opus encode.
OPUS_FILE_TYPE = 102


def _connect():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )


def get_original_file_key(sound_fragment_id: str) -> str | None:
    """Resolve the original (non-opus) Hetzner file_key for a SoundFragment from
    _files. Excludes opus encodes; prefers file_type 101 over legacy 0."""
    conn = _connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT file_key FROM _files "
                "WHERE parent_id = %s AND archived = 0 AND file_type <> %s "
                "ORDER BY file_type DESC LIMIT 1",
                (sound_fragment_id, OPUS_FILE_TYPE),
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def save_analysis(sound_fragment_id: str, metadata: dict) -> int:
    """Write the analysis result into the SoundFragment's `add_info` jsonb column.
    Returns the number of rows updated (0 if the id doesn't exist)."""
    conn = _connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                f"UPDATE {SF_TABLE} SET add_info = %s WHERE id = %s",
                (json.dumps(metadata), sound_fragment_id),
            )
            return cur.rowcount
    finally:
        conn.close()
