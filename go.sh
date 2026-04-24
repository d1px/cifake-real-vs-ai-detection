#!/bin/bash
set -e
PROJECT="/home/dip/cifake-project"

echo "=== CIFAKE DEMO LAUNCHER ==="

# Load virtual camera module if not already loaded
if ! lsmod | grep -q v4l2loopback; then
    echo "Loading virtual camera module..."
    sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="VirtualCam" exclusive_caps=1
else
    echo "Virtual camera module already loaded"
fi

# Check /dev/video10 exists
if [ ! -e /dev/video10 ]; then
    echo "ERROR: /dev/video10 not found. Check v4l2loopback installation."
    exit 1
fi

echo "Virtual camera ready at /dev/video10"

# Kill any existing stream process
pkill -f stream_to_virtual.py 2>/dev/null || true
sleep 1

# Start the stream in background
echo "Starting camera stream in background..."
$PROJECT/venv/bin/python $PROJECT/stream_to_virtual.py &
STREAM_PID=$!
echo "Stream PID: $STREAM_PID"

# Wait for stream to initialise
sleep 3
echo "Stream ready"

# Launch the CIFAKE detector using virtual camera
echo "Launching CIFAKE detector..."
$PROJECT/venv/bin/python $PROJECT/start.py

# When camera.py exits, kill the stream
echo "Cleaning up..."
kill $STREAM_PID 2>/dev/null || true
echo "Done"
