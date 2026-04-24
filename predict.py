import sys
import cv2
import numpy as np
import tensorflow as tf

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH = 'models/cifake_model.h5'
IMG_SIZE   = 32

# ── Validate CLI argument ───────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python predict.py <path_to_image>")
    sys.exit(1)

image_path = sys.argv[1]

# ── Load the trained model ──────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# ── Load and preprocess the image (same pipeline as training) ───────────────────
img = cv2.imread(image_path)
if img is None:
    print(f"Error: could not read image at '{image_path}'")
    sys.exit(1)

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype('float32') / 255.0          # normalise [0, 1]
img = np.expand_dims(img, axis=0)            # add batch dimension → (1, 32, 32, 3)

# ── Run inference ───────────────────────────────────────────────────────────────
prob = model.predict(img, verbose=0)[0][0]   # sigmoid output in [0, 1]

# ── Interpret result (REAL=0, FAKE=1) ──────────────────────────────────────────
if prob >= 0.5:
    label      = "FAKE (AI-Generated)"
    confidence = prob * 100
else:
    label      = "REAL"
    confidence = (1 - prob) * 100

print(f"\nResult     : {label}")
print(f"Confidence : {confidence:.2f}%")
