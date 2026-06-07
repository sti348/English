import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent
WHISPER_EXE = Path(r"D:\Programs\VideoCaptioner\resource\bin\Faster-Whisper-XXL\faster-whisper-xxl.exe")
MODEL_DIR = Path(r"D:\Programs\VideoCaptioner\AppData\models")
MODEL_NAME = "large-v2"
JOBS = {}
JOBS_LOCK = threading.Lock()


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


def extract_whisper_words(whisper_data):
    words = []
    for segment in whisper_data.get("segments", []):
        for item in segment.get("words", []):
            raw = str(item.get("word", "")).strip()
            if not raw:
                continue
            words.append(
                {
                    "word": re.sub(r"\s+", " ", raw),
                    "start": round(float(item["start"]), 3),
                    "end": round(float(item["end"]), 3),
                }
            )
    return words


def build_transcript_payload(whisper_data):
    words = extract_whisper_words(whisper_data)
    passage_segments = []
    current_paragraph = []
    current_sentence = []
    last_end = None

    def flush_sentence():
        nonlocal current_sentence, current_paragraph
        if not current_sentence:
            return
        chunks = []
        for idx in range(0, len(current_sentence), 6):
            chunks.append(" ".join(item["word"] for item in current_sentence[idx : idx + 6]))
        current_paragraph.append(chunks)
        current_sentence = []

    def flush_paragraph():
        nonlocal current_paragraph
        flush_sentence()
        if current_paragraph:
            passage_segments.append(current_paragraph)
            current_paragraph = []

    for item in words:
        gap = 0 if last_end is None else item["start"] - last_end
        if current_sentence and gap > 1.8:
            flush_sentence()
        if current_paragraph and gap > 3.0:
            flush_paragraph()

        current_sentence.append(item)
        last_end = item["end"]

        sentence_text = item["word"].rstrip()
        if re.search(r"[.!?][\"')\]]*$", sentence_text) and len(current_sentence) >= 3:
            flush_sentence()
            if len(current_paragraph) >= 4:
                flush_paragraph()

    flush_paragraph()

    if not passage_segments and words:
        passage_segments = [[[item["word"] for item in words]]]

    transcript = " ".join(item["word"] for item in words)
    return {
        "passageSegments": passage_segments,
        "timings": [[item["start"], item["end"]] for item in words],
        "words": words,
        "transcript": transcript,
    }


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


def run_whisper(audio_path, output_dir, progress=None):
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
    if progress:
      progress("starting faster-whisper", "Launching faster-whisper-large-v2...")

    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    output_lines = []
    started = time.time()
    while True:
        if process.stdout is None:
            break
        line = process.stdout.readline()
        if line:
            clean = line.strip()
            if clean:
                output_lines.append(clean)
                output_lines = output_lines[-80:]
                if progress:
                    progress("transcribing", clean[-500:])
        elif process.poll() is not None:
            break

        if time.time() - started > 900:
            process.kill()
            raise RuntimeError("Whisper timed out after 15 minutes.")

    return_code = process.wait()
    if return_code != 0:
        detail = "\n".join(output_lines[-20:]).strip()
        raise RuntimeError(detail or f"Whisper exited with code {return_code}.")

    if progress:
        progress("reading result", "Reading faster-whisper JSON result...")
    json_files = list(Path(output_dir).glob("*.json"))
    if not json_files:
        detail = "\n".join(output_lines[-20:]).strip()
        raise RuntimeError(f"Whisper did not write a JSON result. {detail}".strip())
    return json.loads(json_files[0].read_text(encoding="utf-8"))


def build_result(audio_content, filename, target_tokens=None, progress=None):
    suffix = Path(filename or "audio.wav").suffix or ".wav"
    with tempfile.TemporaryDirectory(prefix="shadowing_whisper_") as tmp:
        tmp_path = Path(tmp)
        audio_path = tmp_path / f"upload{suffix}"
        if progress:
            progress("upload received", f"Received audio file ({len(audio_content) / 1024 / 1024:.1f} MB).")
        audio_path.write_bytes(audio_content)
        whisper_data = run_whisper(audio_path, tmp_path / "out", progress)
        if progress:
            progress("building transcript", "Building transcript and word timing...")
        transcript_payload = build_transcript_payload(whisper_data)
        if target_tokens:
            timings, cost = align_words(target_tokens, whisper_data)
        else:
            timings = transcript_payload["timings"]
            cost = 0
        return {
            **transcript_payload,
            "timings": timings,
            "source": "faster-whisper-large-v2",
            "alignmentCost": cost,
        }


def set_job(job_id, **updates):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updatedAt"] = time.time()


def cleanup_jobs():
    cutoff = time.time() - 3600
    with JOBS_LOCK:
        for job_id in list(JOBS):
            job = JOBS[job_id]
            if job.get("updatedAt", job.get("createdAt", 0)) < cutoff:
                del JOBS[job_id]


def start_transcription_job(audio_content, filename, target_tokens=None):
    cleanup_jobs()
    job_id = uuid.uuid4().hex
    now = time.time()
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "message": "Queued transcription job...",
            "createdAt": now,
            "updatedAt": now,
            "result": None,
            "error": None,
        }

    def progress(stage, message):
        set_job(job_id, status="running", stage=stage, message=message)

    def worker():
        try:
            result = build_result(audio_content, filename, target_tokens, progress)
            set_job(
                job_id,
                status="done",
                stage="done",
                message=f"Finished: recognized {len(result.get('timings', []))} words.",
                result=result,
            )
        except Exception as exc:
            set_job(
                job_id,
                status="error",
                stage="failed",
                message="Transcription failed.",
                error=str(exc) or exc.__class__.__name__,
            )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return job_id


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
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/transcribe-align", "/api/transcribe-start"}:
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            parts = parse_multipart(self.rfile.read(length), self.headers.get("Content-Type", ""))
            audio = parts.get("audio")
            token_part = parts.get("tokens")
            if not audio:
                raise ValueError("Expected audio field.")
            target_tokens = None
            if token_part:
                target_tokens = json.loads(token_part["content"].decode("utf-8"))

            if parsed.path == "/api/transcribe-start":
                job_id = start_transcription_job(audio["content"], audio["filename"], target_tokens)
                payload = json.dumps({"jobId": job_id}, ensure_ascii=False).encode("utf-8")
            else:
                result = build_result(audio["content"], audio["filename"], target_tokens)
                payload = json.dumps(result, ensure_ascii=False).encode("utf-8")

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

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/transcribe-progress":
            return super().do_GET()

        job_id = parse_qs(parsed.query).get("id", [""])[0]
        with JOBS_LOCK:
            job = dict(JOBS.get(job_id) or {})

        if not job:
            payload = json.dumps({"status": "missing", "error": "Transcription job was not found."}).encode("utf-8")
            self.send_response(404)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        job["elapsed"] = round(time.time() - job.get("createdAt", time.time()), 1)
        payload = json.dumps(job, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
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
