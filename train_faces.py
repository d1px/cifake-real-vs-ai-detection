"""
train_faces.py — Train a CNN to classify real vs AI-generated faces.

Dataset: 140k Real and Fake Faces (real_vs_fake structure)
  data/faces/train/real/   data/faces/train/fake/
  data/faces/valid/real/   data/faces/valid/fake/

Quick-train mode: uses a 10k subset (5k real + 5k fake) so training
completes in minutes rather than hours. Set SUBSET = None to use the
full dataset.

Key differences from train.py (CIFAR-10 objects):
  - Input resolution: 64×64 (faces need more spatial detail than 32×32)
  - Deeper CNN with BatchNormalization to handle higher-resolution input
  - Dropout at multiple stages to fight overfitting on face patterns
  - Data augmentation: flips, zoom, rotation, shifts (realistic face variation)
  - Early stopping: halts if val_loss stops improving for 5 epochs
  - Outputs saved to outputs/faces/

Usage:
    cd ~/cifake-project
    venv/bin/python train_faces.py
"""

import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

# ── Configuration ──────────────────────────────────────────────────────────────
IMG_SIZE    = 64          # 64×64 — needed for face feature resolution
BATCH_SIZE  = 64
EPOCHS      = 10          # reduced for quick training
TRAIN_DIR   = "data/faces/train"
VALID_DIR   = "data/faces/valid"
MODEL_PATH  = "models/cifake_faces_model.h5"
OUTPUT_DIR  = "outputs/faces"

# Set to None to use the full 100k training set.
# 5000 per class = 10k total — trains in ~5 min instead of hours.
SUBSET_PER_CLASS = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("models",   exist_ok=True)

# ── Build a small subset if requested ─────────────────────────────────────────
if SUBSET_PER_CLASS is not None:
    SUBSET_TRAIN_DIR = "data/faces/train_subset"
    for cls in ("real", "fake"):
        src = os.path.join(TRAIN_DIR, cls)
        dst = os.path.join(SUBSET_TRAIN_DIR, cls)
        if os.path.isdir(dst):
            shutil.rmtree(dst)      # clear any previous subset
        os.makedirs(dst)
        files = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        chosen = random.sample(files, min(SUBSET_PER_CLASS, len(files)))
        for fname in chosen:
            shutil.copy2(os.path.join(src, fname), os.path.join(dst, fname))
    ACTIVE_TRAIN_DIR = SUBSET_TRAIN_DIR
    print(f"Subset created: {SUBSET_PER_CLASS} images per class in '{SUBSET_TRAIN_DIR}'")
else:
    ACTIVE_TRAIN_DIR = TRAIN_DIR
    print(f"Using full training set: '{TRAIN_DIR}'")

# ── Data generators ────────────────────────────────────────────────────────────
# Training generator: augment to help the model generalise across lighting,
# pose, and framing variations common in face datasets.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,       # faces look the same mirrored
    zoom_range=0.15,            # slight zoom simulates distance variation
    rotation_range=12,          # mild rotation simulates head tilt
    width_shift_range=0.1,      # horizontal jitter
    height_shift_range=0.1,     # vertical jitter
    brightness_range=[0.8, 1.2],# lighting variation
)

# Validation generator: only normalise — no augmentation on evaluation data
valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

print("Loading training data ...")
train_gen = train_datagen.flow_from_directory(
    ACTIVE_TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",   # alphabetical: fake=0, real=1
    shuffle=True,
)

print("Loading validation data ...")
valid_gen = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,         # must be False for confusion matrix to align correctly
)

print(f"\n  Class mapping : {train_gen.class_indices}")
print(f"  Train samples : {train_gen.samples}")
print(f"  Valid samples : {valid_gen.samples}\n")

# ── Model ──────────────────────────────────────────────────────────────────────
# Four convolutional blocks, each doubling the filter count.
# BatchNormalization after each conv stabilises training at 64×64 resolution.
# Dropout at increasing rates reduces co-adaptation of features.

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    model = Sequential([

        # ── Block 1: edge / low-level feature detectors ───────────────────────
        Conv2D(32, (3, 3), activation="relu", padding="same",
               input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),           # 64 → 32
        Dropout(0.25),

        # ── Block 2: mid-level features (eyes, nose, mouth regions) ──────────
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),           # 32 → 16
        Dropout(0.25),

        # ── Block 3: higher-level face structure features ─────────────────────
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),           # 16 → 8
        Dropout(0.3),

        # ── Block 4: abstract face-level representations ──────────────────────
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),           # 8 → 4
        Dropout(0.3),

        # ── Classifier head ───────────────────────────────────────────────────
        Flatten(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),                 # heaviest dropout before final decision
        Dense(1, activation="sigmoid"),   # 0 = fake, 1 = real (alphabetical)
    ])
    return model

model = build_model()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# ── Callbacks ──────────────────────────────────────────────────────────────────
# Early stopping: restore the weights from the best epoch so we always keep
# the model with the lowest validation loss, even if later epochs overfit.
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

# Reduce LR when val_loss plateaus — helps squeeze out the last few % accuracy
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1,
)

# ── Training ───────────────────────────────────────────────────────────────────
print(f"\nTraining for up to {EPOCHS} epochs (early stopping patience=5) ...\n")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=valid_gen,
    callbacks=[early_stop, reduce_lr],
)

# ── Save model ─────────────────────────────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")

# ── Plots ──────────────────────────────────────────────────────────────────────
epochs_run = range(len(history.history["accuracy"]))

# Accuracy curve
plt.figure(figsize=(9, 5))
plt.plot(epochs_run, history.history["accuracy"],     label="Train Accuracy")
plt.plot(epochs_run, history.history["val_accuracy"], label="Val Accuracy")
plt.title("Model Accuracy — Faces")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
acc_path = os.path.join(OUTPUT_DIR, "accuracy.png")
plt.savefig(acc_path)
plt.close()
print(f"Saving accuracy curve → {acc_path}")

# Loss curve
plt.figure(figsize=(9, 5))
plt.plot(epochs_run, history.history["loss"],     label="Train Loss")
plt.plot(epochs_run, history.history["val_loss"], label="Val Loss")
plt.title("Model Loss — Faces")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
loss_path = os.path.join(OUTPUT_DIR, "loss.png")
plt.savefig(loss_path)
plt.close()
print(f"Saving loss curve    → {loss_path}")

# ── Evaluate on validation set ─────────────────────────────────────────────────
print("\nEvaluating on validation data ...")
valid_gen.reset()
val_loss, val_acc = model.evaluate(valid_gen, verbose=0)
print(f"\n  Validation Accuracy : {val_acc * 100:.2f}%")
print(f"  Validation Loss     : {val_loss:.4f}")

# ── Confusion matrix ───────────────────────────────────────────────────────────
valid_gen.reset()
preds      = (model.predict(valid_gen, verbose=1) >= 0.5).astype(int).flatten()
true_labels = valid_gen.classes   # 0=fake, 1=real (alphabetical)

cm = confusion_matrix(true_labels, preds)
print("\nClass indices:", valid_gen.class_indices)
print("Confusion matrix:\n", cm)

# Map class indices back to readable names for the plot
class_names = [k for k, v in sorted(valid_gen.class_indices.items(), key=lambda x: x[1])]

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("Confusion Matrix — Faces")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"Saving confusion matrix → {cm_path}")

# ── Classification report ──────────────────────────────────────────────────────
print("\nClassification Report:")
print(classification_report(true_labels, preds, target_names=class_names))
print("Done.")
