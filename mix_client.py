import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid

DEFAULT_URL = "http://localhost:38795"
DEFAULT_EXCERPT_SEC = 40.0
DEFAULT_PREVIEW_SEC = 10.0
EXCERPT_BITRATE = "192k"
PREVIEW_SR = 48000


def duration_sec(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def cut(path: str, start: float, length: float, out_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", path,
         "-vn", "-ac", "2", "-c:a", "libopus", "-b:a", EXCERPT_BITRATE, out_path],
        check=True,
    )


def multipart(fields: dict[str, str], files: dict[str, str]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
    for name, path in files.items():
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                 f'filename="{os.path.basename(path)}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()
        with open(path, "rb") as f:
            body += f.read()
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def request_transition(url: str, a_path: str, a_offset: float, c_path: str, c_offset: float,
                       fmt: str, timeout: float) -> tuple[bytes, dict]:
    body, content_type = multipart(
        {"transition_only": "true", "format": fmt,
         "a_offset_sec": f"{a_offset:.3f}", "c_offset_sec": f"{c_offset:.3f}"},
        {"file_a": a_path, "file_c": c_path},
    )
    req = urllib.request.Request(f"{url.rstrip('/')}/mix", data=body, method="POST",
                                 headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            header = resp.headers.get("X-Mix-Transition")
            clip = resp.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"server returned {e.code}: {e.read().decode(errors='replace')}") from e
    if not header:
        raise SystemExit("response has no X-Mix-Transition header")
    return clip, json.loads(header)


def build_preview(a_path: str, c_path: str, clip_path: str, transition: dict,
                  pre_sec: float, post_sec: float, out_path: str) -> None:
    a_start = transition["a_start_sec"]
    c_resume = transition["c_resume_sec"]
    pre = min(pre_sec, a_start)
    graph = ";".join(
        [f"[{i}:a]aresample={PREVIEW_SR},aformat=channel_layouts=stereo[s{i}]" for i in range(3)]
        + ["[s0][s1][s2]concat=n=3:v=0:a=1[out]"]
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-ss", f"{a_start - pre:.3f}", "-t", f"{pre:.3f}", "-i", a_path,
         "-i", clip_path,
         "-ss", f"{c_resume:.3f}", "-t", f"{post_sec:.3f}", "-i", c_path,
         "-filter_complex", graph, "-map", "[out]", out_path],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mix test client: sends only A's tail and C's head to Spectra /mix and saves the transition clip.")
    parser.add_argument("song_a", help="outgoing song (full file)")
    parser.add_argument("song_c", help="incoming song (full file)")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="mix_out", help="output directory")
    parser.add_argument("--excerpt", type=float, default=DEFAULT_EXCERPT_SEC, help="seconds cut from A's tail and C's head")
    parser.add_argument("--format", default="opus", choices=["opus", "wav"])
    parser.add_argument("--preview", action="store_true", help="also write preview.wav: end of A + clip + start of C")
    parser.add_argument("--preview-sec", type=float, default=DEFAULT_PREVIEW_SEC)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total_a = duration_sec(args.song_a)
    a_offset = max(0.0, total_a - args.excerpt)
    with tempfile.TemporaryDirectory() as tmp:
        excerpt_a = os.path.join(tmp, "a_tail.ogg")
        excerpt_c = os.path.join(tmp, "c_head.ogg")
        cut(args.song_a, a_offset, args.excerpt, excerpt_a)
        cut(args.song_c, 0.0, args.excerpt, excerpt_c)
        sent = os.path.getsize(excerpt_a) + os.path.getsize(excerpt_c)
        full = os.path.getsize(args.song_a) + os.path.getsize(args.song_c)
        print(f"sending {sent / 1e6:.2f} MB (full songs: {full / 1e6:.2f} MB), A offset {a_offset:.1f}s")
        clip, transition = request_transition(args.url, excerpt_a, a_offset, excerpt_c, 0.0, args.format, args.timeout)

    clip_path = os.path.join(args.out, f"clip.{args.format}")
    with open(clip_path, "wb") as f:
        f.write(clip)
    with open(os.path.join(args.out, "transition.json"), "w") as f:
        json.dump(transition, f, indent=2)
    print(f"clip: {clip_path} ({len(clip) / 1e6:.2f} MB)")
    print(json.dumps(transition, indent=2))

    if args.preview:
        preview_path = os.path.join(args.out, "preview.wav")
        build_preview(args.song_a, args.song_c, clip_path, transition,
                      args.preview_sec, args.preview_sec, preview_path)
        print(f"preview: {preview_path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"command failed: {' '.join(map(str, e.cmd))}")
