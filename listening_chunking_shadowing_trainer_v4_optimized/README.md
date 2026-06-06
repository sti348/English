# Listening Chunking + Shadowing Trainer v4 Optimized

This version is optimized for the bundled WAV:

`315193bb-6e18-4d59-808f-8f4cb4441663.wav`

## What changed

- The word highlighting is no longer based on a simple uniform progress ratio.
- The WAV was segmented by detected pauses, then aligned to the passage chunks.
- Each word now has a timestamp range, so highlighting should track this audio more closely.
- The original v3 features remain:
  - chunk marking
  - suggested chunks
  - export chunks
  - built-in read aloud
  - uploaded audio playback
  - shadowing recording
  - mixed audio export
  - mic-only export
  - full text / short sentence / hide text modes

## How to use

Open `listening_chunking_shadowing_trainer_optimized.html`.

The bundled WAV should load automatically if the HTML file and WAV stay in the same folder.

If you open the HTML directly from the folder, shadowing starts source-audio playback without microphone recording so the browser will not show a local-file microphone permission dialog. Use the localhost server below when you need microphone recording and mixed-audio export.

Use `Replay sentence (R)` to play only the current sentence. After it stops, click `Next →` or press `→` to manually continue sentence by sentence. Press `←` for the previous sentence.

Recording auto-save is enabled by default from the localhost page. After you stop shadowing, the mixed-audio and mic-only files download automatically when they are ready.

For microphone recording, some browsers require HTTPS or localhost.

To use faster-whisper word timing for the bundled WAV and for newly uploaded audio, run the included local server:

```powershell
D:\Programs\VideoCaptioner\runtime\python.exe whisper_alignment_server.py
```

Or double-click `open_localhost_trainer.bat` to start the local server and open the trainer automatically.

Then open:

```text
http://localhost:8000/listening_chunking_shadowing_trainer_optimized.html
```

Uploaded audio is sent only to this local server. The server uses:

```text
D:\Programs\VideoCaptioner\AppData\models\faster-whisper-large-v2
```

If you only need static playback without Whisper alignment, you can still run:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000/listening_chunking_shadowing_trainer_optimized.html
```
