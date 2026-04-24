#!/usr/bin/env bash
# launch_demo.sh — One-command launcher for the CIFAKE real-time detector.
#
# What this does:
#   1. Creates the virtual camera (/dev/video10) if it doesn't exist
#   2. Starts virtual_camera.py in the background (mirrors real webcam → /dev/video10)
#   3. Waits 2 seconds for the stream to stabilise
#   4. Launches start.py (the CIFAKE demo)
#   5. Cleans up the background mirror process on exit
#
# Usage:
#   bash ~/cifake-project/launch_demo.sh
#   — or just type: CIFAKE   (if the CIFAKE command is installed)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
VIRTUAL_DEVICE="/dev/video10"
MIRROR_PID=""

# ── Cleanup trap: kill mirror process when the script exits ──────────────────
cleanup() {
    if [ -n "$MIRROR_PID" ] && kill -0 "$MIRROR_PID" 2>/dev/null; then
        echo ""
        echo "==> Stopping virtual camera mirror (PID $MIRROR_PID)..."
        kill "$MIRROR_PID" 2>/dev/null || true
        wait "$MIRROR_PID" 2>/dev/null || true
    fi
    echo "Done."
}
trap cleanup EXIT

# ── Step 1: Create virtual camera device if missing ──────────────────────────
if [ ! -e "$VIRTUAL_DEVICE" ]; then
    echo "==> Virtual camera not found — running setup_virtual_camera.sh ..."
    bash "$SCRIPT_DIR/setup_virtual_camera.sh"
else
    echo "==> Virtual camera $VIRTUAL_DEVICE already exists. ✓"
fi

# ── Step 2: Start virtual_camera.py in the background ───────────────────────
if pgrep -f "virtual_camera.py" > /dev/null 2>&1; then
    echo "==> virtual_camera.py is already running. ✓"
else
    echo "==> Starting virtual camera mirror in background..."
    "$VENV_PYTHON" "$SCRIPT_DIR/virtual_camera.py" &
    MIRROR_PID=$!
    echo "    Mirror PID: $MIRROR_PID"
fi

# ── Step 3: Wait for the stream to start ─────────────────────────────────────
echo "==> Waiting 2 seconds for virtual camera stream to stabilise..."
sleep 2

# ── Step 4: Launch the CIFAKE demo ───────────────────────────────────────────
echo "==> Launching CIFAKE detector..."
echo ""
cd "$SCRIPT_DIR"
"$VENV_PYTHON" "$SCRIPT_DIR/start.py"
