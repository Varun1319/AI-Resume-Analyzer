# training/train.py
import json
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

from resume_analyzer.config import RANKER_MODEL_PATH, TRAINING_DATA_PATH
from resume_analyzer.features.builder import FEATURE_COLUMNS, build_feature_rows
from resume_analyzer.ranking.ranker import heuristic_score

TEST_SIZE = 0.2
RANDOM_STATE = 42


def build_features_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    # Expected columns: jd_text, resume_text, label
    assert "jd_text" in df.columns, "CSV must have 'jd_text' column"
    assert "resume_text" in df.columns, "CSV must have 'resume_text' column"
    assert "label" in df.columns, "CSV must have 'label' column"

    pairs = [(str(jd), str(res)) for jd, res in zip(df["jd_text"], df["resume_text"])]
    rows = build_feature_rows(pairs)

    X = np.array([[feats[col] for col in FEATURE_COLUMNS] for feats in rows], dtype=float)
    y = df["label"].astype(int).to_numpy()
    return X, y


def evaluate(model, X_test, y_test) -> dict:
    """Score the trained ranker and two baselines on the held-out split."""
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    tfidf_only = X_test[:, FEATURE_COLUMNS.index("tfidf_score")]
    heuristic = [
        heuristic_score(dict(zip(FEATURE_COLUMNS, row)))
        for row in X_test
    ]

    return {
        "ranker": {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
        },
        # The baselines output raw similarity scores, not probabilities, so
        # only the threshold-free ROC-AUC is comparable for them.
        "baseline_roc_auc": {
            "tfidf_keyword_match": float(roc_auc_score(y_test, tfidf_only)),
            "weighted_heuristic": float(roc_auc_score(y_test, heuristic)),
        },
    }


def train_and_save_model(csv_path=TRAINING_DATA_PATH, model_path=RANKER_MODEL_PATH):
    print(f"[train] Loading training data from {csv_path}")
    X, y = build_features_from_csv(csv_path)
    print(f"[train] Got {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Gradient Boosting classifier as meta-ranker
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )

    print(f"[train] Training on {len(y_train)} samples, holding out {len(y_test)} for evaluation...")
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    metrics["n_train"] = int(len(y_train))
    metrics["n_test"] = int(len(y_test))
    print("[train] Held-out results:")
    print(json.dumps(metrics, indent=2))

    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)
    print(f"[train] Saved trained ranker to {model_path}")

    metrics_path = os.path.splitext(model_path)[0] + "_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Saved held-out metrics to {metrics_path}")


if __name__ == "__main__":
    train_and_save_model()
