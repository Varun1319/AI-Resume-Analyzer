# ml_model.py
from config import RANKER_MODEL_PATH

from typing import List, Dict
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from bert_model import BertResumeMatcher
from skills_extractor import skill_extractor
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


# Try to load trained ranker model (optional)
MODEL_PATH = str(RANKER_MODEL_PATH)
if os.path.exists(MODEL_PATH):
    try:
        TRAINED_RANKER = joblib.load(MODEL_PATH)
        print(f"[LocalMLModel] Loaded trained ranker from {MODEL_PATH}")
    except Exception as e:
        print(f"[LocalMLModel] Failed to load trained ranker: {e}")
        TRAINED_RANKER = None
else:
    TRAINED_RANKER = None


class LocalMLModel:
    """
    Hybrid AI Model:

    - TF-IDF (keyword matching)
    - BERT (semantic meaning)
    - Skill Match (explicit technical alignment)
    - Experience & Education (profile quality)

    If a trained XGBoost model is present at models/ranker_xgb.pkl,
    it will be used as a meta-ranker on top of these features.
    Otherwise, a heuristic weighted formula is used.
    """
    @staticmethod
    def compute_fit_scores_for_jobs(resume_text: str, job_desc_list: list[str]) -> list[float]:
        scores = []
        for jd in job_desc_list:
            if not jd or not jd.strip():
                scores.append(0.0)
            else:
                s = LocalMLModel.compute_fit_score(jd, resume_text)
            scores.append(s)
        return scores



    bert = BertResumeMatcher()  # loads once (GPU if available)

    @staticmethod
    def _tfidf_scores(job_desc: str, resume_texts: List[str]) -> List[float]:
        vect = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = vect.fit_transform([job_desc] + resume_texts)
        sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
        return [float(s) for s in sims]

    @staticmethod
    def compute_feature_dict(job_desc: str, resume_text: str) -> Dict[str, float]:
        """
        Compute the full feature vector used for ranking.
        Can be used both for scoring and for training data generation.
        """
        # TF-IDF
        tfidf_score = LocalMLModel._tfidf_scores(job_desc, [resume_text])[0]

        # BERT
        bert_score = LocalMLModel.bert.score(job_desc, resume_text)

        # Skills
        skill_score, _, _ = skill_extractor.skill_match(resume_text, job_desc)

        # Experience & Education
        years, edu_level, _ = analyze_resume_experience_education(resume_text)
        exp_score = _normalize_experience(years)
        edu_score = _education_score(edu_level)

        return {
            "tfidf_score": tfidf_score,
            "bert_score": bert_score,
            "skill_score": skill_score,
            "exp_score": exp_score,
            "edu_score": edu_score,
            "exp_years": years,
            "edu_level": float(edu_level),
        }

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
    def _ml_score(feats: Dict[str, float]) -> float:
        """
        Use trained meta-ranker (XGBoost) if available.
        Otherwise fallback to heuristic.
        """
        if TRAINED_RANKER is None:
            return LocalMLModel._heuristic_score(feats)

        # Build feature vector in fixed order
        vec = np.array([[feats[col] for col in FEATURE_COLUMNS]], dtype=float)

        # Assume binary classifier with predict_proba
        try:
            proba = TRAINED_RANKER.predict_proba(vec)[0][1]  # prob of "good match"
            return float(proba)
        except Exception as e:
            print(f"[LocalMLModel] Trained ranker error, using heuristic: {e}")
            return LocalMLModel._heuristic_score(feats)

    @staticmethod
    def compute_fit_score(job_desc: str, resume_text: str) -> float:
        """
        Public API: single-resume score.

        - Computes feature dict
        - Uses trained ML model if present
        - Otherwise uses heuristic formula
        """
        if not job_desc.strip() or not resume_text.strip():
            return 0.0

        feats = LocalMLModel.compute_feature_dict(job_desc, resume_text)
        return LocalMLModel._ml_score(feats)

    @staticmethod
    def compute_fit_scores(job_desc: str, resume_texts: List[str]) -> List[float]:
        """
        Multi-resume scorer. Just applies compute_fit_score for each resume.
        (We keep it simple; vectorization can be added later if needed.)
        """
        if not job_desc.strip() or not resume_texts:
            return [0.0] * len(resume_texts)

        scores: List[float] = []
        for res_text in resume_texts:
            feats = LocalMLModel.compute_feature_dict(job_desc, res_text)
            scores.append(LocalMLModel._ml_score(feats))

        return scores
