"""
setup.py — One-time dependency installer for CIFAKE Real-Time Detector.

Auto-detects your OS and CPU architecture, then installs the correct
packages into the current Python environment. Run this once before
using the project.

Usage:
    python setup.py

Supported platforms:
    macOS Apple Silicon (M1/M2/M3)  — tensorflow-macos + tensorflow-metal
    macOS Intel                      — standard tensorflow
    Windows                          — tensorflow + opencv-python-headless
    Linux                            — tensorflow + opencv-python
"""

import sys
import platform
import subprocess
import os

# ── Colour helpers (ANSI — work on Mac/Linux terminals and Win10+) ─────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def info(msg):    print(f"{CYAN}[setup]{RESET} {msg}")
def success(msg): print(f"{GREEN}[setup] ✓ {msg}{RESET}")
def warn(msg):    print(f"{YELLOW}[setup] ⚠ {msg}{RESET}")
def error(msg):   print(f"{RED}[setup] ✗ {msg}{RESET}", file=sys.stderr)


def detect_platform():
    """Return a string identifying the platform variant."""
    system = platform.system()

    if system == "Darwin":
        # Check CPU architecture to distinguish Apple Silicon from Intel
        machine = platform.machine()
        if machine == "arm64":
            return "mac_silicon"
        else:
            return "mac_intel"

    elif system == "Windows":
        return "windows"

    elif system == "Linux":
        return "linux"

    else:
        return "unknown"


def get_requirements_file(variant):
    """Map platform variant to the matching requirements file."""
    mapping = {
        "mac_silicon": "requirements_mac_silicon.txt",
        "mac_intel":   "requirements_mac_intel.txt",
        "windows":     "requirements_windows.txt",
        "linux":       "requirements_linux.txt",
    }
    return mapping.get(variant)


def check_python_version():
    """Warn if the Python version is outside the tested range."""
    major, minor = sys.version_info[:2]
    if major != 3 or minor < 9:
        warn(f"Python {major}.{minor} detected. This project is tested on "
             "Python 3.9–3.12. Older versions may not work.")
    elif minor > 12:
        warn(f"Python {major}.{minor} detected. Newer versions are untested; "
             "TensorFlow wheels may not be available yet.")
    else:
        info(f"Python {major}.{minor} — OK")


def install(requirements_file):
    """Run pip install -r <file> using the current interpreter."""
    cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
    info(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)

    if result.returncode != 0:
        error("Installation failed. See pip output above for details.")
        sys.exit(1)


def verify_imports():
    """Quick smoke-test: try importing the critical packages."""
    packages = {
        "tensorflow": "TensorFlow",
        "cv2":        "OpenCV",
        "numpy":      "NumPy",
        "pygame":     "pygame",
    }
    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            success(f"{name} imported successfully")
        except ImportError:
            error(f"{name} could not be imported after install — "
                  "check the pip output above")
            all_ok = False
    return all_ok


def main():
    print()
    print(f"{BOLD}{'─' * 52}{RESET}")
    print(f"{BOLD}  CIFAKE Real-Time Detector — Setup{RESET}")
    print(f"{BOLD}{'─' * 52}{RESET}")
    print()

    # 1. Python version check
    check_python_version()

    # 2. Detect platform
    variant = detect_platform()
    platform_labels = {
        "mac_silicon": "macOS Apple Silicon (M1/M2/M3)",
        "mac_intel":   "macOS Intel",
        "windows":     "Windows",
        "linux":       "Linux",
    }
    if variant == "unknown":
        error(f"Unrecognised platform: {platform.system()}. "
              "Install dependencies manually using one of the requirements_*.txt files.")
        sys.exit(1)

    info(f"Detected platform: {platform_labels[variant]}")

    # 3. Locate the requirements file
    req_file = get_requirements_file(variant)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    req_path   = os.path.join(script_dir, req_file)

    if not os.path.isfile(req_path):
        error(f"Requirements file not found: {req_path}")
        error("Make sure all project files are present.")
        sys.exit(1)

    info(f"Using requirements file: {req_file}")
    print()

    # 4. Install
    install(req_path)
    print()

    # 5. Verify
    info("Verifying installed packages ...")
    ok = verify_imports()
    print()

    if ok:
        print(f"{BOLD}{'─' * 52}{RESET}")
        success("Setup complete!")
        print()
        print(f"  Next step: {BOLD}python start.py{RESET}")
        print()
        print("  This will launch the real-time CIFAKE detector using")
        print("  your webcam. Press Q or ESC inside the window to quit.")
        print(f"{BOLD}{'─' * 52}{RESET}")
    else:
        error("Some packages failed to import. Setup may be incomplete.")
        sys.exit(1)


if __name__ == "__main__":
    main()
