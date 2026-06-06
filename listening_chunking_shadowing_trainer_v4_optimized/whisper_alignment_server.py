import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parent
WHISPER_EXE = Path(r"D:\Programs\VideoCaptioner\resource\bin\Faster-Whisper-XXL\faster-whisper-xxl.exe")
MODEL_DIR = Path(r"D:\Programs\VideoCaptioner\AppData\models")
MODEL_NAME = "large-v2"


def normalize_word(word):
    return re.sub(r"[^a-z0-9]", "", word.lower())


def expand_whisper_word(word, start, end):
    raw = word.strip()
    cleaned = re.sub(r"[^a-z0-9.]", "", raw.lower())
    if cleaned in {"10", "ten"}:
        return [{"word": "ten", "start": start, "end": end}]
    if cleaned in {".30", "30", "thirty"}:
        return [{"word": "thirty", "start": start, "end": end}]
    if cleaned in {"doesn't", "doesnt"}:
        mid = (start + end) / 2
        return [
            {"word": "does", "start": start, "end": mid},
            {"word": "not", "start": mid, "end": end},
        ]
    if cleaned == "wi-fi":
        mid = (start + end) / 2
        return [
            {"word": "wi", "start": start, "end": mid},
            {"word": "fi", "start": mid, "end": end},
        ]
    return [{"word": raw, "start": start, "end": end}]


def parse_multipart(body, content_type):
    match = re.search(r"boundary=([^;]+)", content_type)
    if not match:
        raise ValueError("Missing multipart boundary.")
    boundary = match.group(1).strip().strip('"').encode()
    parts = {}
    for chunk in body.split(b"--" + boundary):
        chunk = chunk.strip()
        if not chunk or chunk == b"--":
            continue
        if chunk.endswith(b"--"):
            chunk = chunk[:-2].strip()
        header_bytes, _, payload = chunk.partition(b"\r\n\r\n")
        headers = header_bytes.decode("utf-8", "ignore")
        name_match = re.search(r'name="([^"]+)"', headers)
        if not name_match:
            continue
        name = name_match.group(1)
        filename_match = re.search(r'filename="([^"]*)"', headers)
        payload = payload.rstrip(b"\r\n")
        parts[name] = {
            "filename": filename_match.group(1) if filename_match else None,
            "content": payload,
        }
    return parts


def run_whisper(audio_path, output_dir):
    cmd = [
        str(WHISPER_EXE),
        str(audio_path),
        "-m",
        MODEL_NAME,
        "--model_dir",
        str(MODEL_DIR),
        "-l",
        "en",
        "-d",
        "cpu",
        "--output_format",
        "json",
        "--word_timestamps",
        "true",
        "--vad_filter",
        "false",
        "--beep_off",
        "-o",
        str(output_dir),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        timeout=900,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout[-2000:])
    json_files = list(Path(output_dir).glob("*.json"))
    if not json_files:
        raise RuntimeError("Whisper did not write a JSON result.")
    return json.loads(json_files[0].read_text(encoding="utf-8"))


def align_words(target_tokens, whisper_data):
    words = []
    for segment in whisper_data.get("segments", []):
        for item in segment.get("words", []):
            words.extend(expand_whisper_word(item["word"], float(item["start"]), float(item["end"])))

    target_norm = [normalize_word(token) for token in target_tokens]
    word_norm = [normalize_word(item["word"]) for item in words]
    n = len(target_norm)
    m = len(word_norm)
    inf = 10**9
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    prev = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(n + 1):
        for j in range(m + 1):
            current = dp[i][j]
            if current >= inf:
                continue
            if i < n and j < m:
                cost = 0 if target_norm[i] == word_norm[j] else 2
                if {target_norm[i], word_norm[j]} in [{"used", "use"}, {"wifi", "wi"}, {"wifi", "fi"}]:
                    cost = 1
                if current + cost < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = current + cost
                    prev[i + 1][j + 1] = (i, j, "match")
            if i < n and current + 3 < dp[i + 1][j]:
                dp[i + 1][j] = current + 3
                prev[i + 1][j] = (i, j, "skip_target")
            if j < m and current + 3 < dp[i][j + 1]:
                dp[i][j + 1] = current + 3
                prev[i][j + 1] = (i, j, "skip_word")

    pairs = []
    i, j = n, m
    while i or j:
        item = prev[i][j]
        if item is None:
            break
        pi, pj, action = item
        if action == "match":
            pairs.append((pi, pj))
        i, j = pi, pj
    pairs.reverse()

    timings = [None] * n
    for token_index, word_index in pairs:
        timings[token_index] = [
            round(words[word_index]["start"], 3),
            round(words[word_index]["end"], 3),
        ]

    for idx in range(n):
        if timings[idx] is not None:
            continue
        before = next((k for k in range(idx - 1, -1, -1) if timings[k] is not None), None)
        after = next((k for k in range(idx + 1, n) if timings[k] is not None), None)
        if before is not None and after is not None:
            span_start = timings[before][1]
            span_end = timings[after][0]
            missing = after - before - 1
            width = max(0.05, (span_end - span_start) / max(1, missing))
            pos = idx - before - 1
            timings[idx] = [round(span_start + width * pos, 3), round(span_start + width * (pos + 1), 3)]
        elif before is not None:
            start = timings[before][1]
            timings[idx] = [round(start, 3), round(start + 0.25, 3)]
        elif after is not None:
            end = timings[after][0]
            timings[idx] = [round(max(0, end - 0.25), 3), round(end, 3)]
        else:
            timings[idx] = [0, 0.25]

    return timings, dp[n][m]


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        path = unquote(path.split("?", 1)[0].split("#", 1)[0])
        rel = path.lstrip("/").replace("/", os.sep)
        return str((ROOT / rel).resolve())

    def do_POST(self):
        if self.path != "/api/transcribe-align":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            parts = parse_multipart(self.rfile.read(length), self.headers.get("Content-Type", ""))
            audio = parts.get("audio")
            token_part = parts.get("tokens")
            if not audio or not token_part:
                raise ValueError("Expected audio and tokens fields.")
            target_tokens = json.loads(token_part["content"].decode("utf-8"))

            suffix = Path(audio["filename"] or "audio.wav").suffix or ".wav"
            with tempfile.TemporaryDirectory(prefix="shadowing_whisper_") as tmp:
                tmp_path = Path(tmp)
                audio_path = tmp_path / f"upload{suffix}"
                audio_path.write_bytes(audio["content"])
                whisper_data = run_whisper(audio_path, tmp_path / "out")
                timings, cost = align_words(target_tokens, whisper_data)

            payload = json.dumps(
                {
                    "timings": timings,
                    "source": "faster-whisper-large-v2",
                    "alignmentCost": cost,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as exc:
            payload = str(exc).encode("utf-8", "ignore")
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def guess_type(self, path):
        if path.endswith(".json"):
            return "application/json"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"


if __name__ == "__main__":
    if not WHISPER_EXE.exists():
        raise SystemExit(f"Missing Faster-Whisper-XXL executable: {WHISPER_EXE}")
    if not MODEL_DIR.exists():
        raise SystemExit(f"Missing model directory: {MODEL_DIR}")
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("Listening trainer server: http://localhost:8000/listening_chunking_shadowing_trainer_optimized.html")
    print("Whisper timing endpoint: http://localhost:8000/api/transcribe-align")
    server.serve_forever()
