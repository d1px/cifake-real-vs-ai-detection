# virtual_camera.py - mirrors real webcam to /dev/video10 using pyfakewebcam
# lets camera.py and a video call use the camera at the same time
# run setup_virtual_camera.sh first, then this script

import os
import sys
import cv2

REAL_CAMERA_INDEX = 0
VIRTUAL_DEVICE    = "/dev/video10"


def main():
    # check the virtual device was created by setup script
    if not os.path.exists(VIRTUAL_DEVICE):
        print(
            f"ERROR: Virtual camera device '{VIRTUAL_DEVICE}' does not exist.\n"
            "Please run setup_virtual_camera.sh first:\n"
            "  bash ~/cifake-project/setup_virtual_camera.sh"
        )
        sys.exit(1)

    # check pyfakewebcam is installed
    try:
        import pyfakewebcam
    except ImportError:
        print(
            "ERROR: pyfakewebcam is not installed.\n"
            "Run this command to install it:\n"
            "  venv/bin/pip install pyfakewebcam"
        )
        sys.exit(1)

    # open real webcam
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

    # open virtual camera
    print(f"Opening virtual camera at {VIRTUAL_DEVICE} ...")
    try:
        camera = pyfakewebcam.FakeWebcam(VIRTUAL_DEVICE, width, height)
    except Exception as e:
        print(f"ERROR: Could not open {VIRTUAL_DEVICE} for writing: {e}")
        cap.release()
        sys.exit(1)

    print(f"✓ Mirroring webcam → {VIRTUAL_DEVICE}  (Ctrl+C to stop)\n")

    # mirror loop - read from real, write to virtual
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                continue  # drop frame and try again

            # pyfakewebcam needs rgb not bgr
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            camera.schedule_frame(frame_rgb)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        print("Virtual camera mirror stopped.")


if __name__ == "__main__":
    main()
