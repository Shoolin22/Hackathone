from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS_DIR, CONFUSION_MATRIX_PATH, LABELS_PATH, METRICS_PATH, MODEL_PATH, RANDOM_STATE
from src.data_utils import prepare_paths_and_labels, split_file_paths
from src.features import build_feature_matrix


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(16, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading metadata...')
    file_paths, y, label_encoder = prepare_paths_and_labels()
    x_train_paths, x_test_paths, y_train, y_test = split_file_paths(file_paths, y)

    print('Extracting training features...')
    x_train = build_feature_matrix(x_train_paths)
    print('Extracting test features...')
    x_test = build_feature_matrix(x_test_paths)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced_subsample'
        ))
    ])

    print('Training model...')
    model.fit(x_train, y_train)

    print('Evaluating model...')
    y_pred = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

    print(f'Test Accuracy: {accuracy:.4f}')
    print('\nClassification Report:\n')
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, LABELS_PATH)
    save_confusion_matrix(y_test, y_pred, label_encoder.classes_.tolist(), CONFUSION_MATRIX_PATH)

    metrics = {
        'test_accuracy': accuracy,
        'num_classes': int(len(label_encoder.classes_)),
        'num_train_samples': int(len(x_train_paths)),
        'num_test_samples': int(len(x_test_paths)),
        'classification_report': report_dict,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f'\nSaved model to: {MODEL_PATH}')
    print(f'Saved labels to: {LABELS_PATH}')
    print(f'Saved metrics to: {METRICS_PATH}')
    print(f'Saved confusion matrix to: {CONFUSION_MATRIX_PATH}')


if __name__ == '__main__':
    main()
