#!/usr/bin/env python3
"""
Whisper Audio/Video Transcription — Desktop UI
Run with: python app.py
"""

import os
import sys
import tempfile
import threading
import urllib.parse
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext

import torch

from transcribe import extract_audio, transcribe_audio, save_transcription


MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
DEFAULT_MODEL = "small"

LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Chinese": "zh",
    "Japanese": "ja",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Korean": "ko",
}


def url_to_filename(url: str) -> str:
    """Derive output filename from a URL: last 12 chars of URL-encoded string + .txt"""
    encoded = urllib.parse.quote(url, safe="")
    return encoded[-12:] + ".txt"


def download_url(url: str, dest_path: str, log) -> bool:
    """Download a media file from a URL using yt-dlp."""
    import subprocess
    log(f"Downloading '{url}'...")
    result = subprocess.run(
        ["yt-dlp", "-o", dest_path, "--no-playlist", url],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        log("Download error:\n" + result.stderr.decode())
        return False
    return True


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Whisper Transcription")
        self.resizable(True, True)
        self.minsize(600, 460)
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 4}

        # --- Files row ---
        file_frame = tk.Frame(self)
        file_frame.pack(fill="x", **pad)
        tk.Label(file_frame, text="Files:", width=8, anchor="w").pack(side="left")
        self.file_entry = tk.Entry(file_frame)
        self.file_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(file_frame, text="Browse…", command=self._browse).pack(side="left")

        # --- URL row ---
        url_frame = tk.Frame(self)
        url_frame.pack(fill="x", **pad)
        tk.Label(url_frame, text="URL:", width=8, anchor="w").pack(side="left")
        self.url_entry = tk.Entry(url_frame)
        self.url_entry.pack(side="left", fill="x", expand=True)

        # --- Model + Language row ---
        opt_frame = tk.Frame(self)
        opt_frame.pack(fill="x", **pad)

        tk.Label(opt_frame, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Combobox(
            opt_frame, textvariable=self.model_var,
            values=MODELS, state="readonly", width=12,
        ).pack(side="left", padx=(4, 16))

        tk.Label(opt_frame, text="Language:").pack(side="left")
        self.lang_var = tk.StringVar(value="Auto-detect")
        ttk.Combobox(
            opt_frame, textvariable=self.lang_var,
            values=list(LANGUAGES.keys()), state="readonly", width=14,
        ).pack(side="left", padx=4)

        # --- Buttons row ---
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", **pad)
        self.run_btn = tk.Button(
            btn_frame, text="Transcribe", width=14,
            command=self._start, bg="#1a73e8", fg="white",
        )
        self.run_btn.pack(side="left", padx=(0, 8))
        tk.Button(btn_frame, text="Clear", width=8, command=self._clear).pack(side="left")

        # --- Log area ---
        self.log_box = scrolledtext.ScrolledText(
            self, wrap="word", state="disabled", height=16,
        )
        self.log_box.pack(fill="both", expand=True, padx=10, pady=(4, 10))

    # ------------------------------------------------------------------
    def _browse(self):
        paths = filedialog.askopenfilenames(
            title="Select audio/video files",
            filetypes=[
                ("Media files", "*.mp4 *.mov *.mkv *.avi *.mp3 *.wav *.m4a *.flac *.ogg"),
                ("All files", "*.*"),
            ],
        )
        if paths:
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, ";".join(paths))

    def _clear(self):
        self.file_entry.delete(0, "end")
        self.url_entry.delete(0, "end")
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")

    def _log(self, msg: str):
        """Append a line to the log box (thread-safe via after)."""
        def _write():
            self.log_box.config(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.after(0, _write)

    def _set_running(self, running: bool):
        self.after(0, lambda: self.run_btn.config(
            state="disabled" if running else "normal"
        ))

    # ------------------------------------------------------------------
    def _start(self):
        file_val = self.file_entry.get().strip()
        url_val = self.url_entry.get().strip()

        if not file_val and not url_val:
            self._log("Please select file(s) or enter a URL.")
            return

        model_name = self.model_var.get()
        language = LANGUAGES[self.lang_var.get()]

        self._set_running(True)
        threading.Thread(
            target=self._run,
            args=(file_val, url_val, model_name, language),
            daemon=True,
        ).start()

    def _run(self, file_val: str, url_val: str, model_name: str, language):
        import whisper

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
            self._log(f"Device: {'GPU: ' + gpu_name if gpu_name else 'CPU'}")

            self._log(f"Loading model '{model_name}' (downloads if needed)...")
            model = whisper.load_model(model_name, device=device)
            self._log("Model ready.")

            # Collect jobs: list of (input_path, output_path, is_temp)
            jobs = []

            # Local files
            if file_val:
                for path in file_val.split(";"):
                    path = path.strip()
                    if path:
                        output = os.path.splitext(path)[0] + ".txt"
                        jobs.append((path, output, False))

            # URL
            if url_val:
                output_name = url_to_filename(url_val)
                output_path = os.path.join(os.getcwd(), output_name)
                tmp_dl = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                tmp_dl.close()
                if not download_url(url_val, tmp_dl.name, self._log):
                    os.remove(tmp_dl.name)
                    return
                jobs.append((tmp_dl.name, output_path, True))

            for input_path, output_path, is_temp in jobs:
                self._log(f"\n--- {os.path.basename(input_path)} ---")
                tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_wav.close()
                try:
                    extract_audio(input_path, tmp_wav.name)
                    result = transcribe_audio(model, tmp_wav.name, language=language, device=device)
                    save_transcription(result, output_path)
                    self._log(f"Saved → {output_path}")
                finally:
                    if os.path.exists(tmp_wav.name):
                        os.remove(tmp_wav.name)
                    if is_temp and os.path.exists(input_path):
                        os.remove(input_path)

            self._log("\nDone.")

        except Exception as exc:
            self._log(f"Error: {exc}")
        finally:
            self._set_running(False)


if __name__ == "__main__":
    app = App()
    app.mainloop()
