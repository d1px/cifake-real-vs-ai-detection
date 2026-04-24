#!/bin/bash
set -e
PROJECT="/home/dip/cifake-project"

echo "=== CIFAKE DEMO LAUNCHER ==="

# load the kernel module if it's not loaded yet
if ! lsmod | grep -q v4l2loopback; then
    echo "Loading virtual camera module..."
    sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="VirtualCam" exclusive_caps=1
else
    echo "Virtual camera module already loaded"
fi

# make sure /dev/video10 exists
if [ ! -e /dev/video10 ]; then
    echo "ERROR: /dev/video10 not found. Check v4l2loopback installation."
    exit 1
fi

echo "Virtual camera ready at /dev/video10"

# kill any old stream that's running
pkill -f stream_to_virtual.py 2>/dev/null || true
sleep 1

# start the stream in background
echo "Starting camera stream in background..."
$PROJECT/venv/bin/python $PROJECT/stream_to_virtual.py &
STREAM_PID=$!
echo "Stream PID: $STREAM_PID"

# give it a moment to start up
sleep 3
echo "Stream ready"

# launch the detector
echo "Launching CIFAKE detector..."
$PROJECT/venv/bin/python $PROJECT/start.py

# cleanup when done
echo "Cleaning up..."
kill $STREAM_PID 2>/dev/null || true
echo "Done"
