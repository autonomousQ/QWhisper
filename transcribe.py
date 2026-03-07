#!/usr/bin/env python3
"""
Transcribe audio from a video file using OpenAI's Whisper model.

Usage:
    python transcribe.py <video_file> [--model MODEL] [--output OUTPUT] [--language LANGUAGE]

Example:
    python transcribe.py video.mp4
    python transcribe.py video.mp4 --model medium --output transcript.txt
    python transcribe.py video.mp4 --model large --language en
"""

import argparse
import os
import subprocess
import sys
import tempfile

import torch
import whisper


def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract audio from a video file using FFmpeg."""
    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' not found.", file=sys.stderr)
        sys.exit(1)

    command = [
        "ffmpeg",
        "-y",               # overwrite output without asking
        "-i", video_path,   # input video
        "-vn",              # no video output
        "-acodec", "pcm_s16le",  # PCM 16-bit little-endian audio
        "-ar", "16000",     # 16 kHz sample rate (required by Whisper)
        "-ac", "1",         # mono channel
        audio_path,
    ]

    print(f"Extracting audio from '{video_path}'...")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print("FFmpeg error:", result.stderr.decode(), file=sys.stderr)
        sys.exit(1)

    print(f"Audio extracted to '{audio_path}'.")


def transcribe_audio(
    audio_path: str,
    model_name: str = "base",
    language: str | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Load Whisper model and transcribe audio."""
    print(f"Loading Whisper model '{model_name}' on device '{device}'...")
    model = whisper.load_model(model_name, device=device)

    transcribe_options: dict = {"fp16": device == "cuda"}
    if language:
        transcribe_options["language"] = language

    print("Transcribing audio...")
    result = model.transcribe(audio_path, **transcribe_options)
    return result


def save_transcription(result: dict, output_path: str) -> None:
    """Save transcription text to a file."""
    text = result.get("text", "").strip()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")
    print(f"Transcription saved to '{output_path}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio from a video file using OpenAI Whisper."
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en",
                 "medium", "medium.en", "large", "large-v2", "large-v3"],
        help="Whisper model to use (default: base).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the output text file (default: <video_name>.txt).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language of the audio (e.g. 'en', 'fr'). Auto-detected if omitted.",
    )
    args = parser.parse_args()

    # Determine output path
    output_path = args.output or os.path.splitext(args.video)[0] + ".txt"

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")

    # Extract audio to a temporary WAV file, then transcribe
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        extract_audio(args.video, audio_path)
        result = transcribe_audio(audio_path, model_name=args.model,
                                  language=args.language, device=device)
        save_transcription(result, output_path)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    print("Done.")


if __name__ == "__main__":
    main()
