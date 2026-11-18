"""Run the trained model on one .wav file."""
import sys
import numpy as np
import tensorflow as tf
from config import *
from feature_extraction import load_audio, extract_mfcc


def main(path):
    model = tf.keras.models.load_model(MODEL_DIR / "cough_cnn.keras")
    x = extract_mfcc(load_audio(path))[np.newaxis, ..., np.newaxis]
    probs = model.predict(x, verbose=0)[0]
    print(f"TB likelihood score: {probs[1] * 100:.1f}%")
    print("Pre-screening only. Not a diagnosis. Confirm with medical tests.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict.py path/to/cough.wav")
    main(sys.argv[1])
