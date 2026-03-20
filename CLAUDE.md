# Whisper Audio/Video Transcription — Claude Code Guide

## Project Overview

CLI and desktop GUI (tkinter) that transcribe audio and video files to text using OpenAI's Whisper model with CUDA GPU acceleration. Supports batch processing of multiple files and URL-based media extraction via yt-dlp.

## Setup

Install dependencies:
```bash
pip install git+https://github.com/openai/whisper.git
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Install FFmpeg and add to PATH:
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg

# Windows (then add D:\Program Files\ffmpeg\bin to system PATH)
winget install ffmpeg
```

## Running the Desktop GUI

```bash
python transcribe-ui.py
```

**Layout (top → bottom):**
1. Input box (file paths / URLs, one per line) + Browse / Extract buttons on the right
2. Status log window (scrollable)
3. Controls row: Model | Language | Credit | Clear | Transcribe

**Extract buttons** are `tk.Menubutton` dropdowns:
- Extract MP4 ▼ / Extract WEBM ▼ — quality: Best, 1080p, 720p, 480p, 360p
- Extract MP3 ▼ — quality: Best, 320k, 192k, 128k

**Credit button** opens a modal (About dialog) with version, GPL-3.0 license links, and PayPal donation button.

## Running the CLI Script

```bash
python transcribe.py <file> [file ...] [--model MODEL] [--language LANGUAGE]
```

Examples:
```bash
# Single file
python transcribe.py video.mp4

# Multiple files (video and audio mixed)
python transcribe.py 1.mp4 2.mp4 3.mp3 --model medium

# Specify language (useful for non-English content)
python transcribe.py lecture.mp4 --model medium --language Chinese

# Best accuracy, auto-detect language
python transcribe.py interview.mp4 --model large
```

Each input file produces a `.txt` output with the same base name (e.g. `1.mp4` → `1.txt`).

## Available Models

| Model  | Speed    | Accuracy | Multilingual |
|--------|----------|----------|--------------|
| tiny   | Fastest  | Lowest   | Yes          |
| base   | Fast     | Good     | Yes          |
| small  | Medium   | Better   | Yes          |
| medium | Slower   | High     | Yes          |
| large  | Slowest  | Best     | Yes (best)   |

Default model: `base`

> **Tip:** Use `--model large` for best multilingual support. You can omit `--language` to auto-detect.

## Key Files

- `transcribe.py` — CLI transcription script
- `transcribe-ui.py` — desktop GUI (tkinter)
- `requirements.txt` — Python dependencies
- `.claude/hooks/session-start.sh` — installs dependencies on session start (remote only)
- `.claude/settings.json` — Claude Code hook configuration

## Notes

- CUDA is used automatically when an NVIDIA GPU is available; falls back to CPU
- PyTorch with CUDA 12.4 (`cu124`) is required for Python 3.13 — `cu121` has no Python 3.13 wheels
- The model is loaded once and reused for all files in a batch
- Audio is extracted to a temporary WAV file (16 kHz mono) and deleted after transcription
- Both audio files (mp3, wav, m4a, flac, etc.) and video files (mp4, mkv, avi, etc.) are supported
- Output is saved as `<input_name>.txt` next to each input file
- Elapsed time is printed at the end of the run
- `.mp4` files and `.env` are excluded from git via `.gitignore`
- GUI version string follows `YYMMDD` + revision letter format (e.g. `260320A`)
