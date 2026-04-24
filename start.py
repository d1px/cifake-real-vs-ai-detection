"""
start.py — Cross-platform launcher for the CIFAKE Real-Time Detector.

Performs pre-flight checks (model file, dependencies) and sets the
correct environment variables for your OS before launching camera.py.

Usage:
    python start.py

Quit the detector by pressing Q or ESC inside the pygame window.
"""

import sys
import os
import platform
import subprocess

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def info(msg):    print(f"{CYAN}[start]{RESET} {msg}")
def success(msg): print(f"{GREEN}[start] ✓ {msg}{RESET}")
def warn(msg):    print(f"{YELLOW}[start] ⚠ {msg}{RESET}")
def error(msg):   print(f"{RED}[start] ✗ {msg}{RESET}", file=sys.stderr)


# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "models", "cifake_faces_model.h5")
CAMERA_PATH = os.path.join(SCRIPT_DIR, "camera.py")


def check_model():
    """Abort with a clear message if the trained model file is missing."""
    if not os.path.isfile(MODEL_PATH):
        error("Model file not found:")
        error(f"  Expected: {MODEL_PATH}")
        print()
        print("  To create it, train the model first:")
        print(f"    {BOLD}python train_faces.py{RESET}")
        print()
        print("  Or copy a pre-trained models/cifake_faces_model.h5 into this folder.")
        sys.exit(1)

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    success(f"Model found ({size_mb:.1f} MB)")


def check_dependencies():
    """Verify that all required packages can be imported."""
    required = {
        "tensorflow": "TensorFlow",
        "cv2":        "OpenCV",
        "numpy":      "NumPy",
        "pygame":     "pygame",
    }
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)

    if missing:
        error("Missing dependencies: " + ", ".join(missing))
        print()
        print("  Run setup first:")
        print(f"    {BOLD}python setup.py{RESET}")
        sys.exit(1)

    success("All dependencies present")


def check_camera_script():
    """Make sure camera.py exists alongside this launcher."""
    if not os.path.isfile(CAMERA_PATH):
        error(f"camera.py not found at: {CAMERA_PATH}")
        print("  Ensure all project files are in the same directory.")
        sys.exit(1)
    success("camera.py found")

    if not os.path.exists("/dev/video10"):
        warn("Virtual camera /dev/video10 not found.")
        warn("Run go.sh instead of start.py directly to enable Teams + CIFAKE simultaneously.")
        warn("Falling back to real webcam (camera.py will auto-detect).")


def configure_environment():
    """Set OS-specific environment variables so pygame and OpenCV behave correctly."""
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin":
        # macOS: request camera access via AVFoundation; without this flag
        # OpenCV may silently fail to open the webcam on newer macOS versions.
        os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "0")

        # pygame on macOS needs the SDL video driver set explicitly when
        # running outside a full desktop session (e.g. from a bare terminal).
        os.environ.setdefault("SDL_VIDEODRIVER", "cocoa")

        if machine == "arm64":
            info("Platform: macOS Apple Silicon — Metal GPU enabled if available")
        else:
            info("Platform: macOS Intel")

    elif system == "Windows":
        # SDL on Windows works best with the windows driver for pygame fullscreen.
        # Do NOT set DISPLAY or WAYLAND vars — they confuse SDL on Windows.
        os.environ["SDL_VIDEODRIVER"] = "windows"
        # Hide the console window that pygame sometimes spawns on Windows
        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "0,0")
        info("Platform: Windows — SDL video driver set to 'windows'")

    elif system == "Linux":
        # On Linux, detect whether we are running under Wayland or X11 and set
        # the SDL driver accordingly. pygame 2.x supports both natively.
        wayland_display = os.environ.get("WAYLAND_DISPLAY", "")
        xdg_session     = os.environ.get("XDG_SESSION_TYPE", "").lower()
        display         = os.environ.get("DISPLAY", "")

        if wayland_display or xdg_session == "wayland":
            # Wayland session (Hyprland, Sway, GNOME on Wayland, etc.)
            os.environ["SDL_VIDEODRIVER"] = "wayland"
            info("Platform: Linux — Wayland session detected (SDL_VIDEODRIVER=wayland)")

        elif display:
            # Fallback: X11 / XWayland session
            os.environ["SDL_VIDEODRIVER"] = "x11"
            info(f"Platform: Linux — X11 session detected (DISPLAY={display})")

        else:
            # Neither Wayland nor X11 found — warn but still attempt launch
            warn("No display server detected (WAYLAND_DISPLAY and DISPLAY are unset).")
            warn("pygame may fail to open a window. Make sure you are running "
                 "from a desktop session, not an SSH shell without X forwarding.")

    else:
        warn(f"Unknown platform '{system}' — skipping display driver configuration.")


def launch():
    """Hand off execution to camera.py using the same Python interpreter."""
    print()
    info("Launching CIFAKE Real-Time Detector ...")
    info("Press Q or ESC inside the window to quit.")
    print()

    # Run camera.py as a subprocess using the same Python that is running
    # this launcher, so it inherits the venv and the env vars we just set.
    result = subprocess.run([sys.executable, CAMERA_PATH], cwd=SCRIPT_DIR)

    if result.returncode != 0:
        print()
        error(f"camera.py exited with code {result.returncode}.")
        print("  Check the output above for error details.")
        sys.exit(result.returncode)


def main():
    print()
    print(f"{BOLD}{'─' * 52}{RESET}")
    print(f"{BOLD}  CIFAKE Real-Time Detector — Launcher{RESET}")
    print(f"{BOLD}{'─' * 52}{RESET}")
    print()

    check_model()
    check_dependencies()
    check_camera_script()
    configure_environment()
    launch()


if __name__ == "__main__":
    main()
