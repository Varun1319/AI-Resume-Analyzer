# ranking/explain.py
from __future__ import annotations
from functools import lru_cache
import numpy as np
import pandas as pd

from resume_analyzer.features.builder import FEATURE_COLUMNS, build_feature_rows
from resume_analyzer.ranking.ranker import load_ranker


@lru_cache(maxsize=1)
def _get_explainer():
    """SHAP explainer over the same ranker the app scores with, built on first use."""
    model = load_ranker()
    if model is None:
        return None
    import shap

    return shap.TreeExplainer(model)


def explain_match(jd_text: str, resume_text: str) -> tuple[pd.DataFrame, float]:
    """
    Compute SHAP values for a single JD–resume pair.

    Returns:
        df: DataFrame with columns [feature, value, shap_value, abs_shap]
        base_value: model's expected value (bias term)
    """
    explainer = _get_explainer()
    if explainer is None:
        raise RuntimeError(
            "No trained model/explainer available. "
            "Train the ranker first (python -m training.train)."
        )

    feats = build_feature_rows([(jd_text, resume_text)])[0]

    # Convert to ordered vector
    x = np.array([[feats[col] for col in FEATURE_COLUMNS]], dtype=float)

    shap_vals = explainer.shap_values(x)

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
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[-1])
    else:
        base_value = float(base_value)

    return df, base_value
