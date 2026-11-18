"""Shared settings for the Cough Sense pipeline. Adjust to match your device."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
FEATURE_DIR = ROOT / "data" / "features"
MODEL_DIR = ROOT / "models"

CLASSES = ["non_tb", "tb"]   # label 0 = non_tb, label 1 = tb

SAMPLE_RATE = 16000          # Hz
DURATION = 2.0               # seconds per cough clip (padded / trimmed)
N_MFCC = 20
N_FFT = 512
HOP_LENGTH = 256

EPOCHS = 30
BATCH_SIZE = 16
