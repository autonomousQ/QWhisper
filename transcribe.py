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
import threading
import time

import torch
import whisper


def live_timer(stop_event: threading.Event, prefix: str, done_text: str = "Done") -> None:
    """Print elapsed time in-place until stop_event is set, then print done_text."""
    start = time.time()
    while not stop_event.wait(0.1):
        elapsed = time.time() - start
        sys.stdout.write(f"\r{prefix}{elapsed:.1f}s")
        sys.stdout.flush()
    sys.stdout.write(f"\r{prefix}{done_text}\n")
    sys.stdout.flush()


def run_with_timer(fn, prefix: str, done_text: str = "Done"):
    """Run fn() while showing a live timer on the same line. Returns fn's result."""
    stop_event = threading.Event()
    timer = threading.Thread(target=live_timer, args=(stop_event, prefix, done_text), daemon=True)
    timer.start()
    result = fn()
    stop_event.set()
    timer.join()
    return result


def type_text(text: str, delay: float = 0.01) -> None:
    """Print text word by word."""
    words = text.split(" ")
    for i, word in enumerate(words):
        sys.stdout.write(word + (" " if i < len(words) - 1 else ""))
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()


def transcribe_audio(
    model: whisper.Whisper,
    audio_path: str,
    language: str | None = None,
    device: str = "cpu",
) -> dict:
    """Transcribe audio using a pre-loaded Whisper model (no terminal output)."""
    opts: dict = {"fp16": device == "cuda"}
    if language:
        opts["language"] = language
    return model.transcribe(audio_path, **opts)


def save_transcription(result: dict, output_path: str) -> None:
    """Save transcription text to a file."""
    text = result.get("text", "").strip()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


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

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FFmpeg error:", result.stderr.decode(), file=sys.stderr)
        sys.exit(1)


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
        help="Language of the audio (e.g. 'en', 'Chinese'). Only used with large/turbo models. Auto-detected if omitted.",
    )
    args = parser.parse_args()

    # --language is only supported by large and turbo models
    large_models = {"large", "large-v2", "large-v3", "turbo"}
    if args.language and args.model not in large_models:
        print(f"Warning: --language is only supported for large/turbo models. Ignoring for model '{args.model}'.")
        args.language = None

    print("*** START ***")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_label = f"GPU ({torch.cuda.get_device_name(0)})" if device == "cuda" else "CPU"
    print(f"> Device: {device_label}")

    # Load model
    print(f"> Loading model '{args.model}' (downloads if needed)... ", end="", flush=True)
    model = whisper.load_model(args.model, device=device)
    print("Model ready.")

    start_time = time.time()

    for input_path in args.files:
        output_path = os.path.abspath(os.path.splitext(input_path)[0] + ".txt")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name

        try:
            extract_audio(input_path, audio_path)

            # Transcribe with live timer
            transcribe_options: dict = {"fp16": device == "cuda"}
            if args.language:
                transcribe_options["language"] = args.language

            result = run_with_timer(
                lambda: model.transcribe(audio_path, **transcribe_options),
                prefix=f"> Transcribing '{os.path.basename(input_path)}'... ",
            )

            # Save
            sys.stdout.write(f"> Saving to {output_path}... ")
            sys.stdout.flush()
            text = result.get("text", "").strip()
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            sys.stdout.write("Done\n")
            sys.stdout.flush()

            # Type out transcript
            print("> Transcript:")
            type_text(text)

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    elapsed = time.time() - start_time
    print(f"*** DONE (Time elapsed: {elapsed:.0f}s) ***")


if __name__ == "__main__":
    main()
