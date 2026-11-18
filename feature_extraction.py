"""Convert .wav cough recordings into MFCC feature arrays."""
import numpy as np
import librosa
from config import *


def load_audio(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)          # remove silence
    target = int(SAMPLE_RATE * DURATION)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target]


def extract_mfcc(y):
    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return mfcc.T                                       # (time, n_mfcc)


def main():
    X, labels = [], []
    for idx, cls in enumerate(CLASSES):
        files = sorted((RAW_DIR / cls).glob("*.wav"))
        print(f"{cls}: {len(files)} files")
        for f in files:
            X.append(extract_mfcc(load_audio(f)))
            labels.append(idx)
    if not X:
        raise SystemExit("No .wav files found. See data/README.md")
    X = np.array(X)[..., np.newaxis]
    labels = np.array(labels)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(FEATURE_DIR / "features.npz", X=X, y=labels)
    print("Saved", X.shape, "to", FEATURE_DIR / "features.npz")


if __name__ == "__main__":
    main()
