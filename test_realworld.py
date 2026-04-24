# test_realworld.py - test images from a folder
# loads the faces model and classifies every jpg/png it finds
# usage: python test_realworld.py [folder_path]  (defaults to ./test_images/)

import os
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# config
MODEL_PATH          = os.path.join(os.path.dirname(__file__), "models", "cifake_faces_model.h5")
DEFAULT_TEST_FOLDER = os.path.join(os.path.dirname(__file__), "test_images")
IMG_SIZE            = 64
FAKE_THRESHOLD      = 0.5     # sigmoid >= 0.5 means REAL
UNCERTAIN_THRESHOLD = 0.65    # below this = uncertain

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def classify_image(model, image_path):
    # returns (label, confidence_pct) or (None, None) if image can't be loaded
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, None

    # resize, convert to rgb, normalise
    small      = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    small_rgb  = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    normalised = small_rgb.astype("float32") / 255.0
    batch      = np.expand_dims(normalised, axis=0)

    # run the model
    raw_score  = float(model.predict(batch, verbose=0)[0][0])

    if raw_score >= FAKE_THRESHOLD:
        base_label = "REAL"
        confidence = raw_score
    else:
        base_label = "FAKE"
        confidence = 1.0 - raw_score

    confidence_pct = confidence * 100.0

    if confidence < UNCERTAIN_THRESHOLD:
        label = "UNCERTAIN"
    else:
        label = base_label

    return label, confidence_pct


def main():
    # get folder from args or use default
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = DEFAULT_TEST_FOLDER

    folder = os.path.abspath(folder)

    # create it if it doesn't exist yet
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")
        print("Drop .jpg / .jpeg / .png images into that folder, then re-run.")
        sys.exit(0)

    # find all image files
    image_files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        print(f"No jpg/jpeg/png images found in: {folder}")
        print("Drop images into that folder, then re-run.")
        sys.exit(0)

    # load model
    print(f"Loading model from '{MODEL_PATH}' ...")
    model = load_model(MODEL_PATH)
    print("Model loaded.\n")

    # go through each image and classify it
    print(f"{'Filename':<40} {'Prediction':<22} {'Confidence':>10}")
    print("-" * 76)

    count_real      = 0
    count_fake      = 0
    count_uncertain = 0
    count_error     = 0

    for filename in image_files:
        image_path = os.path.join(folder, filename)
        label, conf_pct = classify_image(model, image_path)

        if label is None:
            print(f"{filename:<40} {'ERROR (could not load)':<22} {'':>10}")
            count_error += 1
            continue

        if label == "REAL":
            count_real += 1
        elif label == "FAKE":
            count_fake += 1
        else:
            count_uncertain += 1

        print(f"{filename:<40} {label:<22} {conf_pct:>9.1f}%")

    # print summary at the end
    total = len(image_files)
    print("-" * 76)
    print(f"\nSummary")
    print(f"  Total images tested : {total}")
    print(f"  REAL                : {count_real}")
    print(f"  FAKE                : {count_fake}")
    print(f"  UNCERTAIN           : {count_uncertain}")
    if count_error:
        print(f"  Failed to load      : {count_error}")


if __name__ == "__main__":
    main()
