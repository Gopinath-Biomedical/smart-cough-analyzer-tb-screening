"""Local server: serves the Kaasa IQ web app and runs the trained model.

Usage:
    python code/server.py            # http://localhost:5000  (mic works on localhost)
    python code/server.py --https    # for phones on the same WiFi (self-signed cert,
                                     # needs: pip install cryptography)

Without a trained model (models/cough_cnn.keras) the app still runs, but the
result screen shows "AUDIO MODEL NOT CONNECTED" and no TB score is produced.
"""
import os
import sys
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from config import ROOT, MODEL_DIR
from feature_extraction import load_audio, extract_mfcc
import numpy as np

app = Flask(__name__, static_folder=str(ROOT / "app"), static_url_path="")
model = None
model_path = MODEL_DIR / "cough_cnn.keras"
if model_path.exists():
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    print("Model loaded:", model_path)
else:
    print("WARNING: no trained model at", model_path, "- running without predictions")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(model_loaded=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        request.files["audio"].save(tmp.name)
        x = extract_mfcc(load_audio(tmp.name))[np.newaxis, ..., np.newaxis]
        score = float(model.predict(x, verbose=0)[0][1])   # class 1 = tb
        return jsonify(model_loaded=True, score=score)
    finally:
        os.remove(tmp.name)


if __name__ == "__main__":
    ssl = "adhoc" if "--https" in sys.argv else None
    app.run(host="0.0.0.0", port=5000, ssl_context=ssl)
