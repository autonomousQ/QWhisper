# Whisper Video Transcription — Claude Code Guide

## Project Overview

Python script that transcribes video files to text using OpenAI's Whisper model with CUDA GPU acceleration.

## Setup

Install dependencies:
```bash
pip install git+https://github.com/openai/whisper.git torch
```

Install FFmpeg:
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg

# Windows
winget install ffmpeg
```

## Running the Script

```bash
python transcribe.py <video_file> [--model MODEL] [--output OUTPUT] [--language LANGUAGE]
```

Examples:
```bash
python transcribe.py video.mp4
python transcribe.py lecture.mp4 --model medium
python transcribe.py interview.mp4 --model large --language en --output transcript.txt
```

## Available Models

| Model  | Speed    | Accuracy |
|--------|----------|----------|
| tiny   | Fastest  | Lowest   |
| base   | Fast     | Good     |
| small  | Medium   | Better   |
| medium | Slower   | High     |
| large  | Slowest  | Best     |

Default model: `base`

## Key Files

- `transcribe.py` — main script
- `requirements.txt` — Python dependencies
- `.claude/hooks/session-start.sh` — installs dependencies on session start (remote only)
- `.claude/settings.json` — Claude Code hook configuration

## Notes

- CUDA is used automatically when an NVIDIA GPU is available; falls back to CPU
- Audio is extracted to a temporary WAV file (16 kHz mono) and deleted after transcription
- Output is saved as `<input_name>.txt` next to the input file unless `--output` is specified
