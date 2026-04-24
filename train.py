import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# ── Configuration ──────────────────────────────────────────────────────────────
IMG_SIZE    = 32
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_REAL  = 'data/train/REAL'
TRAIN_FAKE  = 'data/train/FAKE'
TEST_REAL   = 'data/test/REAL'
TEST_FAKE   = 'data/test/FAKE'
MODEL_PATH  = 'models/cifake_model.h5'

# ── Helper: load images from a directory ───────────────────────────────────────
def load_images(directory, label):
    images, labels = [], []
    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        # Resize to 32x32 and convert BGR → RGB
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(label)
    return images, labels

# ── Load training data (REAL=0, FAKE=1) ────────────────────────────────────────
print("Loading training images...")
real_imgs, real_labels = load_images(TRAIN_REAL, label=0)
fake_imgs, fake_labels = load_images(TRAIN_FAKE, label=1)

X_train = np.array(real_imgs + fake_imgs, dtype='float32') / 255.0   # normalise [0, 1]
y_train = np.array(real_labels + fake_labels, dtype='float32')

print(f"  Training samples — REAL: {len(real_imgs)}, FAKE: {len(fake_imgs)}, Total: {len(X_train)}")

# ── Load test data ──────────────────────────────────────────────────────────────
print("Loading test images...")
test_real_imgs, test_real_labels = load_images(TEST_REAL, label=0)
test_fake_imgs, test_fake_labels = load_images(TEST_FAKE, label=1)

X_test = np.array(test_real_imgs + test_fake_imgs, dtype='float32') / 255.0
y_test = np.array(test_real_labels + test_fake_labels, dtype='float32')

print(f"  Test samples     — REAL: {len(test_real_imgs)}, FAKE: {len(test_fake_imgs)}, Total: {len(X_test)}")

# ── Shuffle data so validation split gets both classes ──────────────────────────
indices = np.arange(len(X_train))
np.random.seed(42)
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

# ── Data augmentation ───────────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    validation_split=0.2
)

train_generator = datagen.flow(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    subset='training',
    shuffle=True
)

val_generator = datagen.flow(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    subset='validation',
    shuffle=False
)

# ── Build CNN model ─────────────────────────────────────────────────────────────
print("Building model...")
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),

    # Block 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Block 3
    Conv2D(128, (3, 3), activation='relu'),

    # Classifier head
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.summary()

# ── Compile ─────────────────────────────────────────────────────────────────────
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ── Train ───────────────────────────────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# ── Plot and save accuracy curve ────────────────────────────────────────────────
print("Saving accuracy curve → outputs/accuracy.png")
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/accuracy.png', dpi=150)
plt.close()

# ── Plot and save loss curve ────────────────────────────────────────────────────
print("Saving loss curve → outputs/loss.png")
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/loss.png', dpi=150)
plt.close()

# ── Evaluate on test set ────────────────────────────────────────────────────────
print("\nEvaluating on test data...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Accuracy : {test_acc * 100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")

# ── Generate predictions for classification report ──────────────────────────────
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

# ── Confusion matrix ────────────────────────────────────────────────────────────
print("Saving confusion matrix → outputs/confusion_matrix.png")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['REAL', 'FAKE'],
    yticklabels=['REAL', 'FAKE']
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150)
plt.close()

# ── Classification report ───────────────────────────────────────────────────────
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))

# ── Save model ──────────────────────────────────────────────────────────────────
print(f"Saving model → {MODEL_PATH}")
model.save(MODEL_PATH)
print("Done.")
