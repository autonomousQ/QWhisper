# Whisper Video Transcription (CUDA Enabled)

Transcribe video files into text using OpenAI's Whisper speech-to-text model, running locally with GPU acceleration (CUDA).

The script will:
1. Load a video file
2. Extract the audio automatically
3. Run Whisper transcription
4. Save the output to a `.txt` file

Supported formats: `.mp4`, `.mov`, `.mkv`, `.avi`

---

## Features

- Local transcription (no API required)
- CUDA GPU acceleration
- Simple command-line interface
- Works directly with video files
- Outputs clean `.txt` transcript
- Displays elapsed transcription time

---

## Requirements

- Python 3.13+
- CUDA-compatible NVIDIA GPU (recommended)
- FFmpeg installed and added to PATH
- PyTorch with CUDA

---

## Installation

### 1. Install FFmpeg

**Windows** (via winget)
```bash
winget install ffmpeg
```

**macOS**
```bash
brew install ffmpeg
```

**Linux**
```bash
sudo apt install ffmpeg
```

### 2. Install Python packages

Install Whisper directly from GitHub:
```bash
pip install git+https://github.com/openai/whisper.git
```

Install PyTorch with CUDA support (CUDA 12.4 for Python 3.13):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## Project Structure

```
whisper-video-transcriber/
│
├── transcribe.py
├── README.md
└── videos/
```

---

## Usage

Run the script from the terminal:
```bash
python transcribe.py video.mp4
```

With a larger model:
```bash
python transcribe.py lecture.mp4 --model medium
```

### Model Size Options

| Model  | Speed    | Accuracy | GPU          |
|--------|----------|----------|--------------|
| tiny   | Fastest  | Lowest   | Optional     |
| base   | Fast     | Good     | Optional     |
| small  | Medium   | Better   | Recommended  |
| medium | Slower   | High     | Recommended  |
| large  | Slowest  | Best     | Strongly recommended |
| turbo  | Fast     | High     | Strongly recommended |

> **Note:** `--language` is only available for `large` and `turbo` models. It is ignored for all other models.

---

## Output

For an input file `meeting.mp4`, the script generates `meeting.txt`:

```
Hello everyone. Today we will discuss the quarterly report and project timeline...
```

---

## Optional Improvements

This project can be extended to:
- Generate SRT subtitles
- Batch process multiple videos
- Automatically upload transcripts to YouTube or social media
- Integrate with n8n automation

---

## Reference

- [OpenAI Whisper — GitHub](https://github.com/openai/whisper)
