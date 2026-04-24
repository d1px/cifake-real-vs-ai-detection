# camera.py - live webcam detector using pygame
# reads frames, runs model, shows cyberpunk style UI
# controls: Q/ESC = quit, F = fullscreen, D = face detect toggle, S = screenshot

import os
import sys
import time
import collections
from datetime import datetime

import cv2
import numpy as np
import pygame
from tensorflow.keras.models import load_model

# config
MODEL_PATH          = "models/cifake_faces_model.h5"
IMG_SIZE            = 64
# run go.sh first to get the virtual camera stream going
CAMERA_INDEX        = 10        # /dev/video10, falls back to 0
WINDOW_TITLE        = "CIFAKE Real-Time Detector"
FAKE_THRESHOLD      = 0.5
UNCERTAIN_THRESHOLD = 0.65

# colours for the UI
C_BG      = (10,  10,  15)
C_SB_BG   = (13,  13,  26)
C_CYAN    = (0,   255, 255)
C_MAGENTA = (255, 0,   255)
C_GREEN   = (0,   255, 65)
C_RED     = (255, 0,   51)
C_AMBER   = (255, 170, 0)
C_GRAY    = (120, 120, 140)
C_DARK    = (30,  30,  45)

SIDEBAR_W = 300
STATUS_H  = 28
CONF_H    = 80

# load the model
print(f"Loading model from '{MODEL_PATH}' ...")
model = load_model(MODEL_PATH)
print("Model loaded.\n")

# haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
print("Face detector loaded.\n")

# open camera - try virtual first, fall back to real webcam
print(f"Opening webcam (index {CAMERA_INDEX}) ...")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Virtual camera at index {CAMERA_INDEX} not available — falling back to webcam index 0")
    CAMERA_INDEX = 0
    cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    sys.exit("ERROR: Could not open any camera. Check your webcam is connected.")

ret, probe = cap.read()
if not ret:
    sys.exit("ERROR: Webcam opened but could not read a frame.")
cam_h, cam_w = probe.shape[:2]
print(f"Webcam resolution: {cam_w}×{cam_h}\nPress Q or ESC to quit.\n")

# init pygame
pygame.init()
info          = pygame.display.Info()
scr_w, scr_h  = info.current_w, info.current_h
is_fullscreen = True
screen        = pygame.display.set_mode(
    (scr_w, scr_h), pygame.FULLSCREEN | pygame.RESIZABLE
)
pygame.display.set_caption(WINDOW_TITLE)

font_xl = pygame.font.SysFont("dejavusansmono", 28, bold=True)
font_lg = pygame.font.SysFont("dejavusansmono", 22, bold=True)
font_md = pygame.font.SysFont("dejavusansmono", 16, bold=True)
font_sm = pygame.font.SysFont("dejavusansmono", 14, bold=False)
font_xs = pygame.font.SysFont("dejavusansmono", 12, bold=False)

clock = pygame.time.Clock()

# runtime state
fps           = 0.0
frame_count   = 0
total_frames  = 0
fps_start     = time.time()
session_start = pygame.time.get_ticks()

face_detect_on = True
pred_history   = collections.deque(maxlen=6)

conf_display = 0.0   # smoothed value shown on screen
conf_target  = 0.0   # actual model output

screenshots_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
screenshot_count = 0
flash_msg        = ""
flash_until      = 0

blink_frame = 0
label       = "INITIALIZING"
colour      = C_CYAN
confidence  = 0.0

# helper to draw text and return the height
def txt(surf, text, font, col, x, y):
    s = font.render(text, True, col)
    surf.blit(s, (x, y))
    return s.get_height()

# draws a divider line in the sidebar
def div(surf, sx, y):
    pygame.draw.line(surf, C_CYAN, (sx + 8, y), (sx + SIDEBAR_W - 8, y), 1)
    return y + 10

# draws the whole sidebar panel

def draw_sidebar(surf, sx, win_h):
    pygame.draw.rect(surf, C_SB_BG, (sx, 0, SIDEBAR_W, win_h))
    pygame.draw.line(surf, C_CYAN, (sx, 0), (sx, win_h), 2)

    x = sx + 12
    y = 14

    # header
    s = font_xl.render("CIFAKE DETECTOR", True, C_CYAN)
    surf.blit(s, (sx + (SIDEBAR_W - s.get_width()) // 2, y))
    y += s.get_height() + 2
    s = font_xs.render("v2.0  |  THREAT ANALYSIS SYSTEM", True, C_MAGENTA)
    surf.blit(s, (sx + (SIDEBAR_W - s.get_width()) // 2, y))
    y += s.get_height() + 8
    y = div(surf, sx, y)

    # model info section
    txt(surf, "[ MODEL INFO ]", font_sm, C_CYAN, x, y);  y += 18
    for k, v in [
        ("Model",  os.path.basename(MODEL_PATH)),
        ("Input",  f"{IMG_SIZE}x{IMG_SIZE} px"),
        ("Thresh", f"{UNCERTAIN_THRESHOLD:.2f}"),
        ("Mode",   "FACE CROP" if face_detect_on else "FULL FRAME"),
    ]:
        txt(surf, f"{k}:", font_xs, C_GRAY, x, y)
        txt(surf, v, font_xs, C_CYAN, x + 78, y)
        y += 16
    y += 4;  y = div(surf, sx, y)

    # live stats section
    txt(surf, "[ LIVE STATS ]", font_sm, C_CYAN, x, y);  y += 18
    elapsed = (pygame.time.get_ticks() - session_start) // 1000
    for k, v in [
        ("FPS",     f"{fps:.1f}"),
        ("Frames",  str(total_frames)),
        ("Session", f"{elapsed // 60:02d}:{elapsed % 60:02d}"),
    ]:
        txt(surf, f"{k}:", font_xs, C_GRAY, x, y)
        txt(surf, v, font_xs, C_CYAN, x + 78, y)
        y += 16
    txt(surf, "Status:", font_xs, C_GRAY, x, y);  y += 14
    txt(surf, label, font_md, colour, x, y);       y += 20
    y += 4;  y = div(surf, sx, y)

    # face detect toggle status
    txt(surf, "[ FACE DETECT ]", font_sm, C_CYAN, x, y);  y += 18
    state_col  = C_GREEN if face_detect_on else C_AMBER
    state_text = "ON  — FACE CROP" if face_detect_on else "OFF — FULL FRAME"
    txt(surf, state_text, font_sm, state_col, x, y);       y += 18
    txt(surf, "Press D to toggle", font_xs, C_GRAY, x, y); y += 16
    y += 4;  y = div(surf, sx, y)

    # screenshot section
    txt(surf, "[ SCREENSHOT ]", font_sm, C_CYAN, x, y);           y += 18
    txt(surf, "Press S to save", font_xs, C_GRAY, x, y);           y += 16
    txt(surf, f"Saved this session: {screenshot_count}", font_xs,
        C_CYAN, x, y);                                             y += 16
    y += 4;  y = div(surf, sx, y)

    # keyboard controls
    txt(surf, "[ CONTROLS ]", font_sm, C_CYAN, x, y);  y += 18
    for line in ["Q/ESC  Quit", "F      Fullscreen", "D      Face detect", "S      Screenshot"]:
        txt(surf, line, font_xs, C_GRAY, x, y);  y += 15
    y += 4;  y = div(surf, sx, y)

    # prediction history log
    txt(surf, "[ PRED HISTORY ]", font_sm, C_CYAN, x, y);  y += 18
    for entry in reversed(pred_history):
        if y + 14 > win_h - 6:
            break
        txt(surf, entry["time"],  font_xs, C_GRAY,          x,       y)
        txt(surf, entry["label"], font_xs, entry["colour"],  x + 68,  y)
        txt(surf, f"{entry['confidence'] * 100:.0f}%", font_xs, C_GRAY, x + 148, y)
        y += 15

# top status bar with fps and live indicator

def draw_status_bar(surf, feed_w, blink_on):
    bg = pygame.Surface((feed_w, STATUS_H), pygame.SRCALPHA)
    bg.fill((10, 10, 20, 220))
    surf.blit(bg, (0, 0))
    pygame.draw.line(surf, C_CYAN, (0, STATUS_H - 1), (feed_w, STATUS_H - 1), 1)

    dot_col = C_GREEN if blink_on else C_DARK
    pygame.draw.circle(surf, dot_col, (14, STATUS_H // 2), 5)
    txt(surf, "LIVE", font_xs, C_GREEN, 24, (STATUS_H - 12) // 2)

    ns = font_xs.render(os.path.basename(MODEL_PATH), True, C_GRAY)
    surf.blit(ns, (feed_w // 2 - ns.get_width() // 2, (STATUS_H - ns.get_height()) // 2))

    fs = font_xs.render(f"FPS: {fps:.1f}", True, C_CYAN)
    surf.blit(fs, (feed_w - fs.get_width() - 10, (STATUS_H - fs.get_height()) // 2))

# confidence bar at the bottom

def draw_conf_bar(surf, feed_w, win_h):
    by = win_h - CONF_H
    bg = pygame.Surface((feed_w, CONF_H), pygame.SRCALPHA)
    bg.fill((10, 10, 20, 220))
    surf.blit(bg, (0, by))
    pygame.draw.line(surf, C_CYAN, (0, by), (feed_w, by), 1)

    txt(surf, f"{label}   {conf_display * 100:.1f}%", font_lg, colour, 20, by + 8)

    mg = 20
    ty = by + 44
    th = 14
    tw = feed_w - mg * 2
    pygame.draw.rect(surf, C_DARK, (mg, ty, tw, th), border_radius=6)
    fw = int(tw * min(conf_display, 1.0))
    if fw > 0:
        pygame.draw.rect(surf, colour, (mg, ty, fw, th), border_radius=6)

# main loop
while True:
    now = pygame.time.get_ticks()

    # handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release();  pygame.quit();  sys.exit(0)

        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                print("Quit — closing.")
                cap.release();  pygame.quit();  sys.exit(0)

            if event.key == pygame.K_f:
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    screen = pygame.display.set_mode(
                        (scr_w, scr_h), pygame.FULLSCREEN | pygame.RESIZABLE)
                else:
                    screen = pygame.display.set_mode(
                        (scr_w // 2, scr_h // 2), pygame.RESIZABLE)

            if event.key == pygame.K_d:
                face_detect_on = not face_detect_on
                print(f"Face detection: {'ON' if face_detect_on else 'OFF'}")

            if event.key == pygame.K_s:
                os.makedirs(screenshots_dir, exist_ok=True)
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(screenshots_dir, f"screenshot_{ts}.png")
                pygame.image.save(screen, path)
                screenshot_count += 1
                flash_msg   = "SCREENSHOT SAVED"
                flash_until = now + 1500
                print(f"Saved: {path}")

    # read frame
    ret, frame_bgr = cap.read()
    if not ret:
        continue

    frame_count  += 1
    total_frames += 1
    blink_frame   = (blink_frame + 1) % 60
    blink_on      = blink_frame < 30

    # calculate fps every second
    elapsed_s = time.time() - fps_start
    if elapsed_s >= 1.0:
        fps         = frame_count / elapsed_s
        frame_count = 0
        fps_start   = time.time()

    # run inference
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )
    face_detected = len(faces) > 0

    if face_detect_on and face_detected:
        x_f, y_f, w_f, h_f = max(faces, key=lambda r: r[2] * r[3])
        pad = int(0.3 * min(w_f, h_f))
        x1 = max(0, x_f - pad)
        y1 = max(0, y_f - pad)
        x2 = min(frame_bgr.shape[1], x_f + w_f + pad)
        y2 = min(frame_bgr.shape[0], y_f + h_f + pad)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        crop      = frame_bgr[y1:y2, x1:x2]
        small     = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))  # resize for model
        inp       = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype("float32") / 255.0  # convert to rgb and normalise
        raw_score = float(model.predict(np.expand_dims(inp, 0), verbose=0)[0][0])
        if raw_score >= FAKE_THRESHOLD:
            base_label, base_col, confidence = "REAL", C_GREEN, raw_score
        else:
            base_label, base_col, confidence = "FAKE", C_RED, 1.0 - raw_score
        label  = "UNCERTAIN" if confidence < UNCERTAIN_THRESHOLD else base_label
        colour = C_AMBER    if confidence < UNCERTAIN_THRESHOLD else base_col

    elif not face_detect_on:
        # no face detect, just run the whole frame through
        small     = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE))
        inp       = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        raw_score = float(model.predict(np.expand_dims(inp, 0), verbose=0)[0][0])
        if raw_score >= FAKE_THRESHOLD:
            base_label, base_col, confidence = "REAL", C_GREEN, raw_score
        else:
            base_label, base_col, confidence = "FAKE", C_RED, 1.0 - raw_score
        label  = "UNCERTAIN" if confidence < UNCERTAIN_THRESHOLD else base_label
        colour = C_AMBER    if confidence < UNCERTAIN_THRESHOLD else base_col

    else:
        label, colour, confidence = "NO FACE", C_AMBER, 0.0

    # smooth the confidence bar so it doesn't jump around
    conf_target  = confidence
    conf_display = conf_display + (conf_target - conf_display) * 0.15

    # save to history every 15 frames
    if label not in ("NO FACE", "INITIALIZING") and total_frames % 15 == 0:
        pred_history.append({
            "time":       datetime.now().strftime("%H:%M:%S"),
            "label":      label,
            "confidence": confidence,
            "colour":     colour,
        })

    # figure out layout sizes
    win_w, win_h = screen.get_size()
    feed_w       = win_w - SIDEBAR_W

    # clear background
    screen.fill(C_BG)

    # draw the webcam feed
    frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    cam_surf   = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    feed_vid_h = win_h - STATUS_H - CONF_H
    screen.blit(pygame.transform.scale(cam_surf, (feed_w, feed_vid_h)), (0, STATUS_H))

    # draw overlays
    draw_status_bar(screen, feed_w, blink_on)
    draw_conf_bar(screen, feed_w, win_h)

    # draw sidebar
    draw_sidebar(screen, feed_w, win_h)

    # show screenshot flash message if recently saved
    if now < flash_until:
        fs  = font_xl.render(flash_msg, True, C_CYAN)
        fx  = (win_w - fs.get_width()) // 2
        fy  = (win_h - fs.get_height()) // 2
        pad = 20
        box = pygame.Surface((fs.get_width() + pad * 2, fs.get_height() + pad * 2), pygame.SRCALPHA)
        box.fill((0, 0, 0, 180))
        screen.blit(box, (fx - pad, fy - pad))
        pygame.draw.rect(screen, C_CYAN,
                         (fx - pad, fy - pad, fs.get_width() + pad * 2, fs.get_height() + pad * 2), 2)
        screen.blit(fs, (fx, fy))

    # flip to screen
    pygame.display.flip()
    clock.tick(0)
