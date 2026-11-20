import json

import numpy as np
import pandas as pd

from resume_analyzer.features.builder import FEATURE_COLUMNS
from training import train


def _synthetic_features(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2
    X = rng.normal(0.3 + 0.3 * y[:, None], 0.15, (n, len(FEATURE_COLUMNS)))
    return X, y


def _train(tmp_path, monkeypatch, name="run"):
    X, y = _synthetic_features()
    monkeypatch.setattr(train, "build_features_from_csv", lambda path: (X, y))
    model_path = tmp_path / name / "ranker_gb.pkl"
    train.train_and_save_model("unused.csv", model_path)
    return model_path, json.loads((tmp_path / name / "ranker_gb_metrics.json").read_text())


def test_reports_held_out_metrics_and_baselines(tmp_path, monkeypatch):
    model_path, metrics = _train(tmp_path, monkeypatch)

    assert model_path.exists()
    assert (metrics["n_train"], metrics["n_test"]) == (160, 40)
    assert set(metrics["ranker"]) == {"roc_auc", "accuracy", "precision", "recall", "f1"}
    assert set(metrics["baseline_roc_auc"]) == {"tfidf_keyword_match", "weighted_heuristic"}
    assert metrics["ranker"]["roc_auc"] > 0.8


def test_training_is_reproducible(tmp_path, monkeypatch):
    assert _train(tmp_path, monkeypatch, "a")[1] == _train(tmp_path, monkeypatch, "b")[1]


def test_build_features_from_csv(tmp_path, offline_features):
    csv = tmp_path / "training_data.csv"
    pd.DataFrame({
        "jd_text": ["python developer", "sales lead"],
        "resume_text": ["python engineer", "python engineer"],
        "label": [1, 0],
    }).to_csv(csv, index=False)

    X, y = train.build_features_from_csv(csv)

    assert X.shape == (2, len(FEATURE_COLUMNS))
    assert y.tolist() == [1, 0]
    assert len(offline_features.calls) == 1
