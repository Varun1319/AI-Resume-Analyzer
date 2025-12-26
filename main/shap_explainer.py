# shap_explainer.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import shap
import joblib

from ml_model import FEATURE_COLUMNS, MODEL_PATH, LocalMLModel


# Try to load the trained model and build a global explainer
if os.path.exists(MODEL_PATH):
    MODEL = joblib.load(MODEL_PATH)
    EXPLAINER = shap.TreeExplainer(MODEL)
    print(f"[SHAP] Loaded model from {MODEL_PATH} for explainability.")
else:
    MODEL = None
    EXPLAINER = None
    print("[SHAP] No trained model found. Explainability will not work.")


def explain_single(jd_text: str, resume_text: str) -> tuple[pd.DataFrame, float]:
    """
    Compute SHAP values for a single JD–resume pair.

    Returns:
        df: DataFrame with columns [feature, value, shap_value, abs_shap]
        base_value: model's expected value (bias term)
    """
    if MODEL is None or EXPLAINER is None:
        raise RuntimeError(
            "No trained model/explainer available. "
            "Train the ranker first (python train_ranker.py)."
        )

    # Build feature dict using your existing pipeline
    feats = LocalMLModel.compute_feature_dict(jd_text, resume_text)

    # Convert to ordered vector
    x = np.array([[feats[col] for col in FEATURE_COLUMNS]], dtype=float)

    shap_vals = EXPLAINER.shap_values(x)

    # For tree models, shap_values is (n_samples, n_features)
    if isinstance(shap_vals, list):
        # Some explainers return [values_for_class0, values_for_class1]
        # We'll assume binary classification and take class 1
        contrib = np.array(shap_vals[-1][0])
    else:
        contrib = np.array(shap_vals[0])

    values = [feats[col] for col in FEATURE_COLUMNS]

    df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "value": values,
        "shap_value": contrib,
    })
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False)

    # Expected value (baseline prediction)
    base_value = EXPLAINER.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[-1])
    else:
        base_value = float(base_value)

    return df, base_value
