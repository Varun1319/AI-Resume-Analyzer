# features/builder.py
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_analyzer.features.profile import analyze_resume_experience_education
from resume_analyzer.features.semantic import get_bert_matcher
from resume_analyzer.features.skills import get_skill_extractor

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


def tfidf_score(job_desc: str, resume_text: str) -> float:
    # Fitted per pair, as in training: a shared fit would make each score
    # depend on which other documents were scored alongside it.
    vect = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vect.fit_transform([job_desc, resume_text])
    return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])


def build_feature_rows(pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
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
            "tfidf_score": tfidf_score(job_desc, resume_text),
            "bert_score": bert_score,
            "skill_score": skill_score,
            "exp_score": _normalize_experience(years),
            "edu_score": _education_score(edu_level),
            "exp_years": years,
            "edu_level": float(edu_level),
        })
    return rows
