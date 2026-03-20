#!/usr/bin/env python3
"""
Whisper Audio/Video Transcription — Desktop UI
Run with: python transcribe-ui.py
"""

import os
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
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


_VIDEO_QUALITIES = {
    "Best":  None,
    "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
    "720p":  "bestvideo[height<=720]+bestaudio/best[height<=720]",
    "480p":  "bestvideo[height<=480]+bestaudio/best[height<=480]",
    "360p":  "bestvideo[height<=360]+bestaudio/best[height<=360]",
}

_AUDIO_QUALITIES = {
    "Best": "0",
    "320k": "320K",
    "192k": "192K",
    "128k": "128K",
}

_DIRECT_EXTS = {
    ".mp3", ".mp4", ".wav", ".m4a", ".mkv", ".avi", ".mov",
    ".flac", ".ogg", ".opus", ".webm", ".aac", ".wma",
}


def _is_direct_url(url: str) -> bool:
    """Return True if the URL points directly to an audio/video file."""
    path = urllib.parse.urlparse(url).path
    _, ext = os.path.splitext(path)
    return ext.lower() in _DIRECT_EXTS


def _direct_download(url: str, log) -> str | None:
    """Download a direct audio/video URL with urllib and return the local path."""
    path = urllib.parse.urlparse(url).path
    filename = os.path.basename(path) or "download"
    dest = os.path.join(os.getcwd(), filename)
    log(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        log(f"Download error: {exc}")
        return None
    return dest


def _yt_download(url: str, fmt: str, quality: str, log) -> str | None:
    """
    Download from a URL in the requested format: "mp4", "webm", or "mp3".
    quality is a key from _VIDEO_QUALITIES or _AUDIO_QUALITIES.
    Direct file URLs are fetched with urllib; everything else uses yt-dlp.
    Returns the local file path, or None on failure.
    """
    import subprocess

    if _is_direct_url(url):
        return _direct_download(url, log)

    log(f"Extracting {fmt.upper()} ({quality}) from URL...")

    tmpdir = tempfile.mkdtemp()
    out_template = os.path.join(tmpdir, "%(title)s.%(ext)s")

    base_args = ["-o", out_template, "--no-playlist", url]
    if fmt == "mp3":
        aq = _AUDIO_QUALITIES.get(quality, "0")
        base_args += ["--extract-audio", "--audio-format", "mp3", "--audio-quality", aq]
    else:
        fmt_selector = _VIDEO_QUALITIES.get(quality)
        if fmt_selector:
            base_args += ["-f", fmt_selector]
        base_args += ["--merge-output-format", fmt]

    # Try yt-dlp executable first, fall back to python -m yt_dlp
    for cmd in (["yt-dlp"] + base_args, [sys.executable, "-m", "yt_dlp"] + base_args):
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            break
        except FileNotFoundError:
            continue
    else:
        log("Error: yt-dlp is not installed. Run: pip install yt-dlp")
        return None

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


def _is_url(text: str) -> bool:
    return text.startswith("http://") or text.startswith("https://")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Whisper Transcription")
        self.resizable(True, True)
        self.minsize(640, 480)
        self._timer_len = 0
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 4}

        # --- Top: input text (left) + action buttons (right) ---
        tk.Label(self, text="Files / URLs to transcribe (one per line)",
                 anchor="w").pack(fill="x", padx=10, pady=(8, 0))
        top_frame = tk.Frame(self)
        top_frame.pack(fill="x", **pad)

        self.input_text = tk.Text(top_frame, height=7, wrap="none", undo=True)
        self.input_text.pack(side="left", fill="both", expand=True, padx=(0, 6))

        btn_right = tk.Frame(top_frame)
        btn_right.pack(side="left", fill="y")
        tk.Button(btn_right, text="Browse…", width=15, command=self._browse).pack(
            fill="x", pady=(0, 4)
        )
        for fmt, qualities in [
            ("mp4",  list(_VIDEO_QUALITIES)),
            ("webm", list(_VIDEO_QUALITIES)),
            ("mp3",  list(_AUDIO_QUALITIES)),
        ]:
            mb = tk.Menubutton(
                btn_right, text=f"Extract {fmt.upper()} \U0001f53b",
                width=15, relief="raised", bd=2,
            )
            menu = tk.Menu(mb, tearoff=False)
            mb["menu"] = menu
            for q in qualities:
                menu.add_command(
                    label=q,
                    command=lambda f=fmt, q=q: self._run_extract_urls(f, q),
                )
            mb.pack(fill="x", pady=(0, 4))

        # --- Middle: full-width status log ---
        tk.Label(self, text="Output", anchor="w").pack(fill="x", padx=10, pady=(4, 0))
        self.log_box = scrolledtext.ScrolledText(
            self, wrap="word", state="disabled",
        )
        self.log_box.pack(fill="both", expand=True, padx=10, pady=(4, 4))

        # --- Bottom: model + language + credit + clear + transcribe ---
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
        ).pack(side="left", padx=(4, 16))

        tk.Button(opt_frame, text="Clear", width=8, command=self._clear).pack(side="right")
        self.run_btn = tk.Button(
            opt_frame, text="Transcribe", width=14,
            command=self._start, bg="#1a73e8", fg="white",
        )
        self.run_btn.pack(side="right", padx=(0, 8))
        tk.Button(
            opt_frame, text="Credit", width=8, command=self._show_credit,
        ).pack(side="right", padx=(0, 8))

    # ------------------------------------------------------------------
    def _show_credit(self):
        dlg = tk.Toplevel(self)
        dlg.title("About Whisper Transcription")
        dlg.resizable(False, False)
        dlg.grab_set()

        self.update_idletasks()
        w, h = 360, 400
        x = self.winfo_x() + (self.winfo_width() - w) // 2
        y = self.winfo_y() + (self.winfo_height() - h) // 2
        dlg.geometry(f"{w}x{h}+{x}+{y}")

        f = tk.Frame(dlg, padx=24, pady=20)
        f.pack(fill="both", expand=True)

        tk.Label(f, text="Whisper Transcription",
                 font=("", 16, "bold")).pack()
        tk.Label(f, text="Version 260320A",
                 font=("", 10), fg="#666666").pack(pady=(2, 10))

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=(0, 10))

        tk.Label(f, text="This software is distributed under the\n"
                 "GNU General Public License v3.0 (GPL-3.0).",
                 justify="center", wraplength=300).pack()

        links_frame = tk.Frame(f)
        links_frame.pack(pady=(8, 0))
        for text, url in [
            ("View License", "https://www.gnu.org/licenses/gpl-3.0.en.html"),
            ("GitHub Repo",  "https://github.com/tommykho/whisper"),
        ]:
            lbl = tk.Label(links_frame, text=text, fg="#0078d4", cursor="hand2")
            lbl.pack(side="left", padx=16)
            lbl.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))
            lbl.bind("<Enter>", lambda e, l=lbl: l.config(font=("", 10, "underline")))
            lbl.bind("<Leave>", lambda e, l=lbl: l.config(font=("", 10)))

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=12)

        tk.Label(f, text="If you find this useful, consider buying me a coffee!",
                 justify="center", wraplength=300).pack()
        tk.Button(
            f, text="Donate via PayPal \u2665",
            command=lambda: webbrowser.open("https://paypal.me/tommykho"),
        ).pack(pady=(10, 0))

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=12)

        tk.Button(f, text="Close", width=12, command=dlg.destroy).pack()

    # ------------------------------------------------------------------
    def _get_input_lines(self) -> list[str]:
        """Return non-empty, stripped lines from the input text box."""
        raw = self.input_text.get("1.0", "end").splitlines()
        return [l.strip() for l in raw if l.strip()]

    def _browse(self):
        paths = filedialog.askopenfilenames(
            title="Select audio/video files",
            filetypes=[
                ("Media files", "*.mp4 *.mov *.mkv *.avi *.mp3 *.wav *.m4a *.flac *.ogg"),
                ("All files", "*.*"),
            ],
        )
        for path in paths:
            self._append_input_line(path)

    def _append_input_line(self, text: str):
        """Append a new line to the input text box (thread-safe)."""
        def _write():
            content = self.input_text.get("1.0", "end-1c")
            if content and not content.endswith("\n"):
                self.input_text.insert("end", "\n")
            self.input_text.insert("end", text)
        self.after(0, _write)

    def _replace_url_lines(self, mapping: dict[str, str]):
        """Replace URL lines with their downloaded file paths (thread-safe)."""
        def _write():
            lines = self.input_text.get("1.0", "end").splitlines()
            new_lines = [mapping.get(l.strip(), l) for l in lines]
            self.input_text.delete("1.0", "end")
            self.input_text.insert("1.0", "\n".join(new_lines))
        self.after(0, _write)

    def _clear(self):
        self.input_text.delete("1.0", "end")
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")

    # --- Log helpers (all thread-safe via after) ----------------------

    def _log(self, msg: str):
        def _write():
            self.log_box.config(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.after(0, _write)

    def _log_inline(self, msg: str):
        def _write():
            self.log_box.config(state="normal")
            self.log_box.insert("end", msg)
            self._timer_len = 0
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.after(0, _write)

    def _log_update_inline(self, val: str):
        def _write():
            self.log_box.config(state="normal")
            if self._timer_len > 0:
                self.log_box.delete(f"end - {self._timer_len + 1}c", "end - 1c")
            self.log_box.insert("end", val)
            self._timer_len = len(val)
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.after(0, _write)

    def _log_finish_inline(self, val: str):
        def _write():
            self.log_box.config(state="normal")
            if self._timer_len > 0:
                self.log_box.delete(f"end - {self._timer_len + 1}c", "end - 1c")
            self.log_box.insert("end", val + "\n")
            self._timer_len = 0
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.after(0, _write)

    def _run_with_timer(self, fn, prefix: str, done_text: str = "Done"):
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
        time.sleep(0.15)
        self._log_finish_inline(done_text)
        return result

    def _type_text(self, text: str, delay: int = 60):
        done_event = threading.Event()
        words = text.split(" ")

        def _type_word(i=0):
            self.log_box.config(state="normal")
            if i < len(words):
                chunk = words[i] + (" " if i < len(words) - 1 else "")
                self.log_box.insert("end", chunk)
                self.log_box.see("end")
                self.log_box.config(state="disabled")
                self.after(delay, lambda: _type_word(i + 1))
            else:
                self.log_box.insert("end", "\n")
                self.log_box.see("end")
                self.log_box.config(state="disabled")
                done_event.set()

        self.after(0, _type_word)
        done_event.wait()

    # ------------------------------------------------------------------
    def _set_running(self, running: bool):
        self.after(0, lambda: self.run_btn.config(
            state="disabled" if running else "normal"
        ))

    # ------------------------------------------------------------------
    def _run_extract_urls(self, fmt: str, quality: str = "Best"):
        urls = [l for l in self._get_input_lines() if _is_url(l)]
        if not urls:
            self._log("No URL found in the input. Enter a URL (http/https) on its own line.")
            return
        self._set_running(True)
        threading.Thread(
            target=self._do_extract_urls, args=(urls, fmt, quality), daemon=True
        ).start()

    def _do_extract_urls(self, urls: list[str], fmt: str, quality: str):
        try:
            mapping = {}
            for url in urls:
                dest = _yt_download(url, fmt, quality, self._log)
                if dest:
                    mapping[url] = dest
                    self._log(f"Downloaded → {dest}")
            if mapping:
                self._replace_url_lines(mapping)
        except Exception as exc:
            self._log(f"Error: {exc}")
        finally:
            self._set_running(False)

    # ------------------------------------------------------------------
    def _start(self):
        inputs = self._get_input_lines()
        if not inputs:
            self._log("Please enter file path(s) or URL(s), one per line.")
            return

        model_name = self.model_var.get()
        language = LANGUAGES[self.lang_var.get()]

        self._set_running(True)
        threading.Thread(
            target=self._run,
            args=(inputs, model_name, language),
            daemon=True,
        ).start()

    def _run(self, inputs: list[str], model_name: str, language):
        import whisper

        try:
            self._log("*** START ***")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            device_label = f"GPU ({torch.cuda.get_device_name(0)})" if device == "cuda" else "CPU"
            self._log(f"> Device: {device_label}")

            self._log_inline(f"> Loading model '{model_name}' (downloads if needed)... ")
            model = whisper.load_model(model_name, device=device)
            self._log_finish_inline("Model ready.")

            # Build job list: (input_path, output_path, is_temp)
            jobs = []
            for inp in inputs:
                if _is_url(inp):
                    output_name = url_to_filename(inp)
                    output_path = os.path.abspath(os.path.join(os.getcwd(), output_name))
                    dest = _yt_download(inp, "mp4", "Best", self._log)
                    if dest:
                        jobs.append((dest, output_path, True))
                else:
                    output = os.path.abspath(os.path.splitext(inp)[0] + ".txt")
                    jobs.append((inp, output, False))

            if not jobs:
                self._log("No valid inputs to transcribe.")
                return

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

                    self._log("> Transcript:")
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
