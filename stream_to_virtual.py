"""
stream_to_virtual.py — Mirror /dev/video0 → /dev/video10 (v4l2loopback virtual camera).

Reads frames from the real webcam and writes them to the virtual camera device so that
multiple apps (Microsoft Teams, camera.py) can access the camera simultaneously.

Usage:
    venv/bin/python stream_to_virtual.py

Stop with Ctrl+C.
"""

import cv2
import pyfakewebcam
import signal
import sys

REAL_CAM   = 0
VIRTUAL_CAM = "/dev/video10"
WIDTH      = 1280
HEIGHT     = 720

print("Starting camera stream...")
cap = cv2.VideoCapture(REAL_CAM)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    print("ERROR: Cannot open real webcam at index 0")
    sys.exit(1)

try:
    camera = pyfakewebcam.FakeWebcam(VIRTUAL_CAM, WIDTH, HEIGHT)
except Exception as e:
    print(f"ERROR: Cannot open virtual camera at {VIRTUAL_CAM}")
    print(f"Detail: {e}")
    print("Run go.sh instead of this script directly — it loads the module first.")
    sys.exit(1)


def cleanup(sig, frame):
    print("Stopping stream...")
    cap.release()
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

print(f"Streaming /dev/video0 → {VIRTUAL_CAM} — press Ctrl+C to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    camera.schedule_frame(frame_rgb)
