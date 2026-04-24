#!/usr/bin/env bash
# setup_virtual_camera.sh — Load v4l2loopback and create /dev/video10
# Run this once before launching the CIFAKE demo when a video call is active.
# Requires v4l2loopback-dkms: sudo pacman -S v4l2loopback-dkms linux-headers

set -e

echo "==> Loading v4l2loopback kernel module..."
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="VirtualCam" exclusive_caps=1

echo "==> Virtual camera devices:"
ls /dev/video*

if [ -e /dev/video10 ]; then
    echo ""
    echo "✓ Virtual camera is ready at /dev/video10"
    echo "  You can now run: CIFAKE"
else
    echo ""
    echo "✗ ERROR: /dev/video10 was not created. Check that v4l2loopback-dkms is installed."
    exit 1
fi
