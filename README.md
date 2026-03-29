```
 ██████╗ ██╗    ██╗██╗  ██╗██╗███████╗██████╗ ███████╗██████╗
██╔═══██╗██║    ██║██║  ██║██║██╔════╝██╔══██╗██╔════╝██╔══██╗
██║   ██║██║ █╗ ██║███████║██║███████╗██████╔╝█████╗  ██████╔╝
██║▄▄ ██║██║███╗██║██╔══██║██║╚════██║██╔═══╝ ██╔══╝  ██╔══██╗
╚██████╔╝╚███╔███╔╝██║  ██║██║███████║██║     ███████╗██║  ██║
 ╚══▀▀═╝  ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝
QWhisper AI Audio/Video Extractor & Transcription
```
Transcribe audio and video files to text using OpenAI's Whisper speech-to-text model, running locally. Supports batch processing of multiple files in one command. Automatically uses CUDA GPU acceleration when available, and falls back to CPU otherwise.

The script will:
1. Accept one or more audio/video files
2. Extract the audio automatically
3. Run Whisper transcription
4. Save each output to a `.txt` file with the same base name

Supported formats: `.mp4`, `.mov`, `.mkv`, `.avi`, `.mp3`, `.wav`, `.m4a`, `.flac`, and more

---

## Features

- Local transcription (no API required)
- Batch processing — transcribe multiple files in one command
- Supports both audio and video files
- CUDA GPU acceleration (automatically falls back to CPU if unavailable)
- Simple command-line interface and desktop GUI (`transcribe-ui.py`)
- Outputs clean `.txt` transcript per file
- Displays elapsed transcription time
- GUI: quality-aware video/audio extraction via yt-dlp dropdown menus
- GUI: About/Credit modal with license info and donation link

---

## Requirements

- Python 3.13+
- FFmpeg installed and added to PATH
- CUDA-compatible NVIDIA GPU (recommended, but not required)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AutonomousQ/QWhisper.git
cd QWhisper
```

### 2. Install FFmpeg

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

### 3. Install Python packages

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
QWhisper/
├── transcribe.py        # CLI script
├── transcribe-ui.py     # Desktop GUI (tkinter)
├── requirements.txt
└── README.md
```

---

## Usage

### Desktop GUI

```bash
python transcribe-ui.py
```

- Paste file paths or URLs into the input box (one per line), or click **Browse…**
- Use the **Extract MP4 ▼ / WEBM ▼ / MP3 ▼** dropdown buttons to download media at a chosen quality via yt-dlp
- Select model and language, then click **Transcribe**
- Click **Credit** for version info, license, and donation link

### Command Line

```bash
python transcribe.py <file> [file ...] [--model MODEL] [--language LANGUAGE]
```

Single file:
```bash
python transcribe.py video.mp4
```

Multiple files (video and audio mixed):
```bash
python transcribe.py 1.mp4 2.mp4 3.mp3 --model medium
```

Specify language (large/turbo models only):
```bash
python transcribe.py lecture.mp4 --model large --language Chinese
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

## License

This code is released under [The GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Support

If you find this useful, consider buying me a coffee!

[![Donate via PayPal](https://img.shields.io/badge/Donate-PayPal-blue?logo=paypal)](https://paypal.me/tommykho)

