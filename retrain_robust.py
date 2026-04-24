"""
retrain_robust.py — Fine-tune cifake_faces_model.h5 for webcam robustness.

Strategy:
  - Load existing model (preserves learned features)
  - Freeze first 8 layers (keep low-level features intact)
  - Aggressive augmentation (simulates webcam lighting/colour conditions)
  - Low learning rate (0.0001) — fine-tuning, not full retraining
  - Save as models/cifake_faces_model_v2.h5 (never overwrites original)
  - Auto-updates camera.py only if val_acc > 88.7%
"""

import os
import time
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info spam

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR   = os.path.join(BASE_DIR, "data", "faces", "train")
VALID_DIR   = os.path.join(BASE_DIR, "data", "faces", "valid")
SRC_MODEL   = os.path.join(BASE_DIR, "models", "cifake_faces_model.h5")
DST_MODEL   = os.path.join(BASE_DIR, "models", "cifake_faces_model_v2.h5")
CAMERA_PY   = os.path.join(BASE_DIR, "camera.py")

# ── Hyper-parameters ───────────────────────────────────────────────────────────
IMG_SIZE         = 64
BATCH_SIZE       = 32
EPOCHS           = 8
LR               = 1e-4
SUBSET_PER_CLASS = 8000
BASELINE_ACC     = 0.887

# ── Load model ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("CIFAKE Robust Fine-Tune")
print("=" * 60)
print(f"\nLoading base model from: {SRC_MODEL}")
model = load_model(SRC_MODEL)
print(f"Model loaded. Total layers: {len(model.layers)}")
print(f"Output shape: {model.output_shape}")

# ── Freeze first 8 layers ──────────────────────────────────────────────────────
FREEZE_UP_TO = 8
for i, layer in enumerate(model.layers):
    layer.trainable = i >= FREEZE_UP_TO

frozen  = sum(1 for l in model.layers if not l.trainable)
trained = sum(1 for l in model.layers if l.trainable)
print(f"Frozen layers: {frozen}  |  Trainable layers: {trained}")

# ── Recompile with low LR ──────────────────────────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# ── Aggressive augmentation (webcam simulation) ────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.5, 1.5],
    channel_shift_range=30.0,
    shear_range=0.15,
    fill_mode="nearest",
)

valid_gen = ImageDataGenerator(rescale=1.0 / 255)

train_flow = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=42,
)

valid_flow = valid_gen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

# ── Subset to SUBSET_PER_CLASS images per class ────────────────────────────────
total_train = SUBSET_PER_CLASS * 2
total_valid = min(len(valid_flow.filenames), 4000)
steps_per_epoch  = total_train  // BATCH_SIZE
validation_steps = total_valid  // BATCH_SIZE

print(f"\nTraining on {total_train} samples ({SUBSET_PER_CLASS} per class)")
print(f"Validating on {total_valid} samples")
print(f"Steps/epoch: {steps_per_epoch}  |  LR: {LR}  |  Epochs: {EPOCHS}")
print(f"Augmentation: flip, rotation±20°, zoom±25%, brightness [0.5–1.5], channel_shift 30\n")

# ── Callbacks ──────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ── Train ──────────────────────────────────────────────────────────────────────
print("Starting fine-tuning ...")
t0 = time.time()

history = model.fit(
    train_flow,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=valid_flow,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1,
)

elapsed = time.time() - t0
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)

# ── Results ────────────────────────────────────────────────────────────────────
best_val_acc = max(history.history["val_accuracy"])
final_val_acc = history.history["val_accuracy"][-1]
report_acc    = best_val_acc   # EarlyStopping restores best weights

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Training time     : {minutes}m {seconds}s")
print(f"Best val accuracy : {report_acc * 100:.2f}%")
print(f"Baseline accuracy : {BASELINE_ACC * 100:.1f}%")
print(f"Improvement       : {(report_acc - BASELINE_ACC) * 100:+.2f}%")

# ── Save new model ─────────────────────────────────────────────────────────────
model.save(DST_MODEL)
print(f"\nNew model saved to: {DST_MODEL}")

# ── Update camera.py if improved ──────────────────────────────────────────────
if report_acc > BASELINE_ACC:
    with open(CAMERA_PY, "r") as f:
        src = f.read()
    updated = re.sub(
        r'MODEL_PATH\s*=\s*"models/cifake_faces_model\.h5"',
        'MODEL_PATH          = "models/cifake_faces_model_v2.h5"',
        src,
    )
    if updated != src:
        with open(CAMERA_PY, "w") as f:
            f.write(updated)
        print(f"\ncamera.py UPDATED → now uses cifake_faces_model_v2.h5")
    else:
        print("\nWARN: MODEL_PATH line not found in camera.py — manual update needed.")
else:
    print(
        f"\nNew model ({report_acc * 100:.2f}%) did NOT beat baseline ({BASELINE_ACC * 100:.1f}%)."
        "\nOriginal model kept — camera.py unchanged."
    )

print("\nDone.")
