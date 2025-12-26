# train_ranker.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib

from ml_model import LocalMLModel, FEATURE_COLUMNS, MODEL_PATH


def build_features_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    # Expected columns: jd_text, resume_text, label
    assert "jd_text" in df.columns, "CSV must have 'jd_text' column"
    assert "resume_text" in df.columns, "CSV must have 'resume_text' column"
    assert "label" in df.columns, "CSV must have 'label' column"

    X_list = []
    y_list = []

    for _, row in df.iterrows():
        jd = str(row["jd_text"])
        res = str(row["resume_text"])
        label = int(row["label"])

        feats = LocalMLModel.compute_feature_dict(jd, res)
        X_list.append([feats[col] for col in FEATURE_COLUMNS])
        y_list.append(label)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)
    return X, y


def train_and_save_model(csv_path: str, model_path: str = MODEL_PATH):
    print(f"[train_ranker] Loading training data from {csv_path}")
    X, y = build_features_from_csv(csv_path)
    print(f"[train_ranker] Got {X.shape[0]} samples, {X.shape[1]} features")

    # Gradient Boosting classifier as meta-ranker
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
    )

    print("[train_ranker] Training model...")
    model.fit(X, y)

    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)
    print(f"[train_ranker] Saved trained ranker to {model_path}")


if __name__ == "__main__":
    csv_path = "training_data.csv"
    train_and_save_model(csv_path)
