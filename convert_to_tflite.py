"""Quantize the trained model to int8 TFLite and emit a C array for ESP32."""
import numpy as np
import tensorflow as tf
from config import *


def main():
    model = tf.keras.models.load_model(MODEL_DIR / "cough_cnn.keras")
    X = np.load(FEATURE_DIR / "features.npz")["X"].astype(np.float32)

    def representative():
        for i in range(min(100, len(X))):
            yield [X[i:i + 1]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite = conv.convert()

    out = MODEL_DIR / "cough_model.tflite"
    out.write_bytes(tflite)
    print(f"Saved {out}  ({len(tflite) / 1024:.1f} KB)")

    # C header for the ESP32 firmware (same as `xxd -i`)
    hex_bytes = ", ".join(f"0x{b:02x}" for b in tflite)
    (MODEL_DIR / "cough_model_data.h").write_text(
        f"alignas(8) const unsigned char cough_model[] = {{{hex_bytes}}};\n"
        f"const unsigned int cough_model_len = {len(tflite)};\n"
    )
    print("Saved C header: models/cough_model_data.h")


if __name__ == "__main__":
    main()
