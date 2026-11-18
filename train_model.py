"""Train a small CNN on MFCC features and report clinically relevant metrics."""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from config import *


def build_model(input_shape):
    # Kept small so it can be quantized and run on an ESP32
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])


def main():
    data = np.load(FEATURE_DIR / "features.npz")
    X, y = data["X"], data["y"]
    # NOTE: if you have multiple clips per person, split by person, not by clip,
    # otherwise the test score will be over-optimistic.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model(X.shape[1:])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(X_tr, y_tr, validation_split=0.2, epochs=EPOCHS,
              batch_size=BATCH_SIZE)

    pred = np.argmax(model.predict(X_te), axis=1)
    print(classification_report(y_te, pred, target_names=CLASSES))
    cm = confusion_matrix(y_te, pred)
    print("Confusion matrix:\n", cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"Sensitivity: {tp / (tp + fn):.3f}  Specificity: {tn / (tn + fp):.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "cough_cnn.keras")
    np.savez(MODEL_DIR / "test_split.npz", X=X_te, y=y_te)


if __name__ == "__main__":
    main()
