from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from src.config import LABELS_PATH, MODEL_PATH
from src.features import extract_feature_vector


def predict_audio(file_path: Path) -> tuple[str, float]:
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError(
            'Trained model not found. Run train.py first so the model and labels are saved in artifacts/.'
        )

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABELS_PATH)

    feature = extract_feature_vector(file_path).reshape(1, -1)

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(feature)[0]
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
    else:
        predicted_index = int(model.predict(feature)[0])
        confidence = 0.0

    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_label, confidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict audio event from an audio file.')
    parser.add_argument('audio_file', type=str, help='Path to the audio file')
    args = parser.parse_args()

    label, confidence = predict_audio(Path(args.audio_file))
    print(f'Predicted event: {label}')
    print(f'Confidence: {confidence:.2%}')
