"""
virtual_camera.py — Mirror the real webcam to /dev/video10 (v4l2loopback).

This script reads frames from the physical webcam (index 0) and writes them
to the virtual camera device at /dev/video10 so that TWO applications can
access the camera at the same time:
  - Your video call uses /dev/video0  (real webcam)
  - camera.py uses /dev/video10      (virtual mirror)

Usage:
    cd ~/cifake-project
    venv/bin/python virtual_camera.py

Prerequisites:
    1. Run setup_virtual_camera.sh first to create /dev/video10
    2. pip install pyfakewebcam  (inside the venv)

Stop with Ctrl+C.
"""

import os
import sys
import cv2

REAL_CAMERA_INDEX = 0
VIRTUAL_DEVICE    = "/dev/video10"


def main():
    # ── Check virtual device exists ───────────────────────────────────────────
    if not os.path.exists(VIRTUAL_DEVICE):
        print(
            f"ERROR: Virtual camera device '{VIRTUAL_DEVICE}' does not exist.\n"
            "Please run setup_virtual_camera.sh first:\n"
            "  bash ~/cifake-project/setup_virtual_camera.sh"
        )
        sys.exit(1)

    # ── Import pyfakewebcam (give a clear message if missing) ─────────────────
    try:
        import pyfakewebcam
    except ImportError:
        print(
            "ERROR: pyfakewebcam is not installed.\n"
            "Run this command to install it:\n"
            "  venv/bin/pip install pyfakewebcam"
        )
        sys.exit(1)

    # ── Open real webcam ──────────────────────────────────────────────────────
    print(f"Opening real webcam at index {REAL_CAMERA_INDEX} ...")
    cap = cv2.VideoCapture(REAL_CAMERA_INDEX)
    if not cap.isOpened():
        print(
            f"ERROR: Could not open webcam at index {REAL_CAMERA_INDEX}.\n"
            "Check that a camera is connected."
        )
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {width}x{height}")

    # ── Open virtual camera for writing ──────────────────────────────────────
    print(f"Opening virtual camera at {VIRTUAL_DEVICE} ...")
    try:
        camera = pyfakewebcam.FakeWebcam(VIRTUAL_DEVICE, width, height)
    except Exception as e:
        print(f"ERROR: Could not open {VIRTUAL_DEVICE} for writing: {e}")
        cap.release()
        sys.exit(1)

    print(f"✓ Mirroring webcam → {VIRTUAL_DEVICE}  (Ctrl+C to stop)\n")

    # ── Mirror loop ───────────────────────────────────────────────────────────
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                continue  # drop frame and try again

            # pyfakewebcam expects RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            camera.schedule_frame(frame_rgb)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        print("Virtual camera mirror stopped.")


if __name__ == "__main__":
    main()
