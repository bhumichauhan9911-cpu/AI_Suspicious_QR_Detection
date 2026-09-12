
"""
Easy Deep Learning part of the project.

This trains a small CNN using generated QR-like image patterns.
It is intentionally simple for a student project demonstration.

Class 0 = Safe
Class 1 = Suspicious
"""

from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
MODEL = BASE / "models" / "qr_cnn.keras"

rng = np.random.default_rng(42)
images, labels = [], []

# Generate simple synthetic QR-like images.
# Safe images: clean QR-style square patterns.
# Suspicious images: noisy/altered patterns.
for label in [0, 1]:
    for _ in range(300):
        img = np.ones((64, 64), dtype=np.uint8) * 255

        # random QR-like blocks
        for _ in range(35):
            x = int(rng.integers(2, 60))
            y = int(rng.integers(2, 60))
            s = int(rng.integers(2, 7))
            value = 0 if rng.random() > 0.25 else 255
            cv2.rectangle(img, (x, y), (min(63, x+s), min(63, y+s)), int(value), -1)

        # finder patterns
        for x, y in [(4,4), (44,4), (4,44)]:
            cv2.rectangle(img, (x,y), (x+15,y+15), 0, -1)
            cv2.rectangle(img, (x+3,y+3), (x+12,y+12), 255, -1)
            cv2.rectangle(img, (x+6,y+6), (x+9,y+9), 0, -1)

        if label == 1:
            # suspicious class has more image noise/distortion
            noise = rng.normal(0, 35, (64,64))
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        images.append(img.astype("float32") / 255.0)
        labels.append(label)

X = np.array(images).reshape(-1,64,64,1)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cnn = models.Sequential([
    layers.Input(shape=(64,64,1)),
    layers.Conv2D(16, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

cnn.fit(X_train, y_train, epochs=5, batch_size=32,
        validation_split=0.2, verbose=1)

loss, acc = cnn.evaluate(X_test, y_test, verbose=0)
print("CNN test accuracy:", round(float(acc), 4))

MODEL.parent.mkdir(exist_ok=True)
cnn.save(MODEL)
print("CNN model saved:", MODEL)
