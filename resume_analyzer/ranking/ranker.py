# ranking/ranker.py
"""
Match scoring over the hybrid features:

- TF-IDF (keyword matching)
- BERT (semantic meaning)
- Skill Match (explicit technical alignment)
- Experience & Education (profile quality)

If a trained ranker is present at models/ranker_gb.pkl, it is used as a
meta-ranker on top of these features. Otherwise, a heuristic weighted
formula is used.
"""
import os
from functools import lru_cache
from typing import Dict, List, Tuple

import joblib
import numpy as np

from resume_analyzer.config import RANKER_MODEL_PATH
from resume_analyzer.features.builder import FEATURE_COLUMNS, build_feature_rows


@lru_cache(maxsize=1)
def load_ranker():
    """Trained ranker, loaded once on first use; None if it is missing or unreadable."""
    if not os.path.exists(RANKER_MODEL_PATH):
        return None
    try:
        model = joblib.load(RANKER_MODEL_PATH)
        print(f"[ranker] Loaded trained ranker from {RANKER_MODEL_PATH}")
        return model
    except Exception as e:
        print(f"[ranker] Failed to load trained ranker: {e}")
        return None


def heuristic_score(feats: Dict[str, float]) -> float:
    """
    Fallback scoring formula when no trained ranker is available.
    """
    # Text similarity = mix of tfidf + bert
    text_score = 0.4 * feats["tfidf_score"] + 0.6 * feats["bert_score"]

    # Heuristic final:
    # final = 0.6 * text + 0.2 * skills + 0.1 * exp + 0.1 * edu
    final_score = (
        0.6 * text_score +
        0.2 * feats["skill_score"] +
        0.1 * feats["exp_score"] +
        0.1 * feats["edu_score"]
    )
    return float(final_score)


def score_features(rows: List[Dict[str, float]]) -> List[float]:
    """
    Use the trained ranker if available (one batched prediction).
    Otherwise fall back to the heuristic.
    """
    if not rows:
        return []

    ranker = load_ranker()
    if ranker is not None:
        X = np.array([[r[col] for col in FEATURE_COLUMNS] for r in rows], dtype=float)
        try:
            return [float(p) for p in ranker.predict_proba(X)[:, 1]]  # prob of "good match"
        except Exception as e:
            print(f"[ranker] Trained ranker error, using heuristic: {e}")

    return [heuristic_score(r) for r in rows]


def score_pairs(pairs: List[Tuple[str, str]]) -> List[float]:
    """Match score for each (job_desc, resume) pair; a pair with a blank side scores 0."""
    scores = [0.0] * len(pairs)
    live = [i for i, (jd, res) in enumerate(pairs) if jd.strip() and res.strip()]
    rows = build_feature_rows([pairs[i] for i in live])
    for i, score in zip(live, score_features(rows)):
        scores[i] = score
    return scores


def score_pair(job_desc: str, resume_text: str) -> float:
    return score_pairs([(job_desc, resume_text)])[0]
