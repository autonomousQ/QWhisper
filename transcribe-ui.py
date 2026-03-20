#!/usr/bin/env python3
"""
Whisper Audio/Video Transcription — Desktop UI
Run with: python transcribe-ui.py
"""

import os
import tempfile
import threading
import time
import urllib.parse
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext

import torch

from transcribe import extract_audio


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


def _yt_download(url: str, audio_only: bool, log) -> str | None:
    """
    Download video or audio-only from a URL via yt-dlp.
    Saves to CWD and returns the resulting file path, or None on failure.
    """
    import subprocess

    kind = "audio" if audio_only else "video"
    log(f"Extracting {kind} from URL...")

    tmpdir = tempfile.mkdtemp()
    out_template = os.path.join(tmpdir, "%(title)s.%(ext)s")

    cmd = ["yt-dlp", "-o", out_template, "--no-playlist", url]
    if audio_only:
        cmd += ["--extract-audio", "--audio-format", "mp3"]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        log("yt-dlp error:\n" + result.stderr.decode())
        return None

    files = os.listdir(tmpdir)
    if not files:
        log("Error: yt-dlp produced no output file.")
        return None

    src = os.path.join(tmpdir, files[0])
    dest = os.path.join(os.getcwd(), files[0])
    os.rename(src, dest)
    os.rmdir(tmpdir)
    return dest


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Whisper Transcription")
        self.resizable(True, True)
        self.minsize(640, 480)
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
        self.url_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(
            url_frame, text="Extract Video \U0001f53b",
            command=self._extract_video,
        ).pack(side="left", padx=(0, 4))
        tk.Button(
            url_frame, text="Extract Audio \U0001f53b",
            command=self._extract_audio,
        ).pack(side="left")

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

    # --- Log helpers (all thread-safe via after) ----------------------

    def _log(self, msg: str):
        """Append a line to the log box."""
        def _write():
            self.log_box.config(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.after(0, _write)

    def _log_inline(self, msg: str):
        """Append text without a trailing newline, set mark for later updates."""
        def _write():
            self.log_box.config(state="normal")
            self.log_box.insert("end", msg)
            self.log_box.mark_set("status_val", "end")
            self.log_box.mark_gravity("status_val", "left")
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.after(0, _write)

    def _log_update_inline(self, val: str):
        """Overwrite the value portion after the status_val mark (no newline)."""
        def _write():
            self.log_box.config(state="normal")
            self.log_box.delete("status_val", "end")
            self.log_box.insert("end", val)
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.after(0, _write)

    def _log_finish_inline(self, val: str):
        """Finalize the inline status line with val and a newline."""
        def _write():
            self.log_box.config(state="normal")
            self.log_box.delete("status_val", "end")
            self.log_box.insert("end", val + "\n")
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.after(0, _write)

    def _run_with_timer(self, fn, prefix: str, done_text: str = "Done"):
        """
        Run fn() (blocking, call from worker thread) while showing a live
        elapsed-time counter in the log box. Returns fn's result.
        """
        stop_event = threading.Event()
        start_time = time.time()

        self._log_inline(prefix)

        def _tick():
            if stop_event.is_set():
                return
            elapsed = time.time() - start_time
            self._log_update_inline(f"{elapsed:.1f}s")
            self.after(100, _tick)

        self.after(100, _tick)

        result = fn()

        stop_event.set()
        time.sleep(0.15)  # let any pending tick see the flag before we finalize
        self._log_finish_inline(done_text)
        return result

    def _type_text(self, text: str, delay: int = 10):
        """
        Type text character-by-character into the log box.
        Blocks the calling (worker) thread until typing is complete.
        """
        done_event = threading.Event()

        def _type_char(i=0):
            self.log_box.config(state="normal")
            if i < len(text):
                self.log_box.insert("end", text[i])
                self.log_box.see("end")
                self.log_box.config(state="disabled")
                self.after(delay, lambda: _type_char(i + 1))
            else:
                self.log_box.insert("end", "\n")
                self.log_box.see("end")
                self.log_box.config(state="disabled")
                done_event.set()

        self.after(0, _type_char)
        done_event.wait()

    # ------------------------------------------------------------------
    def _set_running(self, running: bool):
        self.after(0, lambda: self.run_btn.config(
            state="disabled" if running else "normal"
        ))

    def _add_to_files(self, path: str):
        """Append a path to the Files entry (thread-safe)."""
        def _write():
            current = self.file_entry.get().strip()
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, (current + ";" + path) if current else path)
        self.after(0, _write)

    # ------------------------------------------------------------------
    def _extract_video(self):
        url = self.url_entry.get().strip()
        if not url:
            self._log("Please enter a URL first.")
            return
        self._set_running(True)
        threading.Thread(
            target=self._run_extract, args=(url, False), daemon=True
        ).start()

    def _extract_audio(self):
        url = self.url_entry.get().strip()
        if not url:
            self._log("Please enter a URL first.")
            return
        self._set_running(True)
        threading.Thread(
            target=self._run_extract, args=(url, True), daemon=True
        ).start()

    def _run_extract(self, url: str, audio_only: bool):
        try:
            dest = _yt_download(url, audio_only, self._log)
            if dest:
                self._add_to_files(dest)
                self._log(f"Downloaded → {dest}")
        except Exception as exc:
            self._log(f"Error: {exc}")
        finally:
            self._set_running(False)

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
            self._log("*** START ***")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            device_label = f"GPU ({torch.cuda.get_device_name(0)})" if device == "cuda" else "CPU"
            self._log(f"> Device: {device_label}")

            model = self._run_with_timer(
                lambda: whisper.load_model(model_name, device=device),
                prefix=f"> Loading model '{model_name}' (downloads if needed)... ",
                done_text="Model ready.",
            )

            # Collect jobs: list of (input_path, output_path, is_temp)
            jobs = []

            if file_val:
                for path in file_val.split(";"):
                    path = path.strip()
                    if path:
                        output = os.path.abspath(os.path.splitext(path)[0] + ".txt")
                        jobs.append((path, output, False))

            if url_val:
                output_name = url_to_filename(url_val)
                output_path = os.path.abspath(os.path.join(os.getcwd(), output_name))
                dest = _yt_download(url_val, False, self._log)
                if not dest:
                    return
                jobs.append((dest, output_path, True))

            start_time = time.time()

            for input_path, output_path, is_temp in jobs:
                tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_wav.close()
                try:
                    extract_audio(input_path, tmp_wav.name)

                    transcribe_opts: dict = {"fp16": device == "cuda"}
                    if language:
                        transcribe_opts["language"] = language

                    wav_path = tmp_wav.name
                    result = self._run_with_timer(
                        lambda: model.transcribe(wav_path, **transcribe_opts),
                        prefix=f"> Transcribing '{os.path.basename(input_path)}'... ",
                    )

                    self._log_inline(f"> Saving to {output_path}... ")
                    text = result.get("text", "").strip()
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(text + "\n")
                    self._log_finish_inline("Done")

                    self._log("Transcript:")
                    self._type_text(text)

                finally:
                    if os.path.exists(tmp_wav.name):
                        os.remove(tmp_wav.name)
                    if is_temp and os.path.exists(input_path):
                        os.remove(input_path)

            elapsed = time.time() - start_time
            self._log(f"*** DONE (Time elapsed: {elapsed:.0f}s) ***")

        except Exception as exc:
            self._log(f"Error: {exc}")
        finally:
            self._set_running(False)


if __name__ == "__main__":
    app = App()
    app.mainloop()
