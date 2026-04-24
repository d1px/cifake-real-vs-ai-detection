#!/usr/bin/env bash
# launch_demo.sh - starts the virtual camera then launches the detector
# usage: bash ~/cifake-project/launch_demo.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
VIRTUAL_DEVICE="/dev/video10"
MIRROR_PID=""

# kill mirror process when script exits
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

# step 1: create virtual camera if needed
if [ ! -e "$VIRTUAL_DEVICE" ]; then
    echo "==> Virtual camera not found — running setup_virtual_camera.sh ..."
    bash "$SCRIPT_DIR/setup_virtual_camera.sh"
else
    echo "==> Virtual camera $VIRTUAL_DEVICE already exists. ✓"
fi

# step 2: start the mirror in background if not already running
if pgrep -f "virtual_camera.py" > /dev/null 2>&1; then
    echo "==> virtual_camera.py is already running. ✓"
else
    echo "==> Starting virtual camera mirror in background..."
    "$VENV_PYTHON" "$SCRIPT_DIR/virtual_camera.py" &
    MIRROR_PID=$!
    echo "    Mirror PID: $MIRROR_PID"
fi

# step 3: wait a moment for stream to start
echo "==> Waiting 2 seconds for virtual camera stream to stabilise..."
sleep 2

# step 4: launch the detector
echo "==> Launching CIFAKE detector..."
echo ""
cd "$SCRIPT_DIR"
"$VENV_PYTHON" "$SCRIPT_DIR/start.py"
