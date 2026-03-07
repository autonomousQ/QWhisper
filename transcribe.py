#!/usr/bin/env python3
"""
Transcribe audio/video files using OpenAI's Whisper model.

Usage:
    python transcribe.py <file> [file ...] [--model MODEL] [--language LANGUAGE]

Example:
    python transcribe.py video.mp4
    python transcribe.py 1.mp4 2.mp4 3.mp3 --model medium --language Chinese
    python transcribe.py lecture.mp4 --model large
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

import torch
import whisper


def extract_audio(input_path: str, audio_path: str) -> None:
    """Extract/convert audio from a video or audio file using FFmpeg."""
    if not os.path.isfile(input_path):
        print(f"Error: File '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    command = [
        "ffmpeg",
        "-y",                    # overwrite output without asking
        "-i", input_path,        # input file (video or audio)
        "-vn",                   # no video output
        "-acodec", "pcm_s16le",  # PCM 16-bit little-endian audio
        "-ar", "16000",          # 16 kHz sample rate (required by Whisper)
        "-ac", "1",              # mono channel
        audio_path,
    ]

    print(f"Extracting audio from '{input_path}'...")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print("FFmpeg error:", result.stderr.decode(), file=sys.stderr)
        sys.exit(1)


def transcribe_audio(
    model: whisper.Whisper,
    audio_path: str,
    language: str | None = None,
    device: str = "cpu",
) -> dict:
    """Transcribe audio using a pre-loaded Whisper model."""
    transcribe_options: dict = {"fp16": device == "cuda"}
    if language:
        transcribe_options["language"] = language

    print("Transcribing...")
    result = model.transcribe(audio_path, **transcribe_options)
    return result


def save_transcription(result: dict, output_path: str) -> None:
    """Save transcription text to a file."""
    text = result.get("text", "").strip()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")
    print(f"Saved to '{output_path}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video files using OpenAI Whisper."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Input audio or video file(s).",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en",
                 "medium", "medium.en", "large", "large-v2", "large-v3"],
        help="Whisper model to use (default: base). Use 'large' for best multilingual support.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language of the audio (e.g. 'en', 'Chinese'). Auto-detected if omitted.",
    )
    args = parser.parse_args()

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU.")

    # Load model once for all files
    print(f"Loading Whisper model '{args.model}' on '{device}'...")
    model = whisper.load_model(args.model, device=device)

    start_time = time.time()

    for input_path in args.files:
        print(f"\n--- Processing '{input_path}' ---")
        output_path = os.path.splitext(input_path)[0] + ".txt"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name

        try:
            extract_audio(input_path, audio_path)
            result = transcribe_audio(model, audio_path, language=args.language, device=device)
            save_transcription(result, output_path)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    elapsed = time.time() - start_time
    print(f"\nDone. Time elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
