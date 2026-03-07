#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo "Installing Python dependencies..."
pip install --quiet openai-whisper torch

echo "Checking ffmpeg..."
if ! command -v ffmpeg &>/dev/null; then
  apt-get update -qq
  apt-get install -y -q --fix-missing ffmpeg || {
    echo "Warning: ffmpeg install encountered errors. Attempting minimal install..."
    apt-get install -y -q --no-install-recommends ffmpeg || true
  }
fi

if command -v ffmpeg &>/dev/null; then
  echo "ffmpeg is available: $(ffmpeg -version 2>&1 | head -1)"
else
  echo "Warning: ffmpeg could not be installed. Audio extraction will fail at runtime."
fi

echo "Session start setup complete."
