# ml_model.py
from config import RANKER_MODEL_PATH

from functools import lru_cache
from typing import List, Dict, Tuple
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from bert_model import get_bert_matcher
from skills_extractor import get_skill_extractor
from experience_education import analyze_resume_experience_education


# Order of features used for training & inference
FEATURE_COLUMNS = [
    "tfidf_score",
    "bert_score",
    "skill_score",
    "exp_score",
    "edu_score",
    "exp_years",
    "edu_level",
]


def _normalize_experience(years: float) -> float:
    if years <= 0:
        return 0.0
    if years >= 5:
        return 1.0
    return float(years / 5.0)


def _education_score(level: int) -> float:
    mapping = {
        0: 0.0,  # Unknown
        1: 0.3,  # Diploma / 12th
        2: 0.6,  # Bachelor's
        3: 0.8,  # Master's
        4: 1.0,  # PhD
    }
    return mapping.get(level, 0.0)


MODEL_PATH = str(RANKER_MODEL_PATH)


@lru_cache(maxsize=1)
def load_ranker():
    """Trained ranker, loaded once on first use; None if it is missing or unreadable."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[LocalMLModel] Loaded trained ranker from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"[LocalMLModel] Failed to load trained ranker: {e}")
        return None


class LocalMLModel:
    """
    Hybrid AI Model:

    - TF-IDF (keyword matching)
    - BERT (semantic meaning)
    - Skill Match (explicit technical alignment)
    - Experience & Education (profile quality)

    If a trained ranker is present at models/ranker_gb.pkl,
    it is used as a meta-ranker on top of these features.
    Otherwise, a heuristic weighted formula is used.
    """

    @staticmethod
    def _tfidf_score(job_desc: str, resume_text: str) -> float:
        # Fitted per pair, as in training: a shared fit would make each score
        # depend on which other documents were scored alongside it.
        vect = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = vect.fit_transform([job_desc, resume_text])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])

    @staticmethod
    def compute_feature_rows(pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """
        Feature dicts for many (job_desc, resume) pairs, with one batched BERT call.
        Used for both scoring and training, so the model always sees the same features.
        """
        if not pairs:
            return []

        bert_scores = get_bert_matcher().pair_scores(pairs)
        skills = get_skill_extractor()

        rows = []
        for (job_desc, resume_text), bert_score in zip(pairs, bert_scores):
            skill_score, _, _ = skills.skill_match(resume_text, job_desc)
            years, edu_level, _ = analyze_resume_experience_education(resume_text)
            rows.append({
                "tfidf_score": LocalMLModel._tfidf_score(job_desc, resume_text),
                "bert_score": bert_score,
                "skill_score": skill_score,
                "exp_score": _normalize_experience(years),
                "edu_score": _education_score(edu_level),
                "exp_years": years,
                "edu_level": float(edu_level),
            })
        return rows

    @staticmethod
    def compute_feature_dict(job_desc: str, resume_text: str) -> Dict[str, float]:
        return LocalMLModel.compute_feature_rows([(job_desc, resume_text)])[0]

    @staticmethod
    def _heuristic_score(feats: Dict[str, float]) -> float:
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

    @staticmethod
    def _ml_scores(rows: List[Dict[str, float]]) -> List[float]:
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
                print(f"[LocalMLModel] Trained ranker error, using heuristic: {e}")

        return [LocalMLModel._heuristic_score(r) for r in rows]

    @staticmethod
    def score_pairs(pairs: List[Tuple[str, str]]) -> List[float]:
        """Match score for each (job_desc, resume) pair; a pair with a blank side scores 0."""
        scores = [0.0] * len(pairs)
        live = [i for i, (jd, res) in enumerate(pairs) if jd.strip() and res.strip()]
        rows = LocalMLModel.compute_feature_rows([pairs[i] for i in live])
        for i, score in zip(live, LocalMLModel._ml_scores(rows)):
            scores[i] = score
        return scores

    @staticmethod
    def compute_fit_score(job_desc: str, resume_text: str) -> float:
        return LocalMLModel.score_pairs([(job_desc, resume_text)])[0]

    @staticmethod
    def compute_fit_scores(job_desc: str, resume_texts: List[str]) -> List[float]:
        """Score many resumes against one job description."""
        return LocalMLModel.score_pairs([(job_desc, r) for r in resume_texts])

    @staticmethod
    def compute_fit_scores_for_jobs(resume_text: str, job_desc_list: List[str]) -> List[float]:
        """Score one resume against many job descriptions."""
        return LocalMLModel.score_pairs([(jd or "", resume_text) for jd in job_desc_list])
