# experience_education.py

from typing import Tuple
from functools import lru_cache
import re


def extract_years_experience(text: str) -> float:
    """
    Extracts years of experience from free-form text using regex heuristics.

    Examples it catches:
    - "3 years of experience"
    - "2 yrs experience"
    - "5+ years"
    - "4 yr"

    Returns:
        float years (0.0 for fresher / not found)
    """
    if not text:
        return 0.0

    text_low = text.lower()

    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|year|yrs|yr)\s+of\s+experience",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|year|yrs|yr)\b",
    ]

    values = []

    for pat in patterns:
        matches = re.findall(pat, text_low)
        for m in matches:
            try:
                val = float(m)
                if 0.0 <= val <= 50.0:  # sanity bound
                    values.append(val)
            except ValueError:
                pass

    # Fresher handling
    if "fresher" in text_low or "entry level" in text_low:
        values.append(0.0)

    if not values:
        return 0.0

    return float(max(values))


def extract_education_level(text: str) -> Tuple[int, str]:
    """
    Returns:
        (level, label)
        level:
            0 = Unknown
            1 = Diploma / 12th
            2 = Bachelor's
            3 = Master's
            4 = PhD
    """
    if not text:
        return 0, "Unknown"

    t = text.lower()

    # PhD
    if "phd" in t or "doctor of philosophy" in t:
        return 4, "PhD"

    # Master's
    masters_keywords = [
        "master of science", "master of technology",
        "m.tech", "mtech", "m.sc", "msc", "mca", "ms "
    ]
    if any(k in t for k in masters_keywords):
        return 3, "Master's"

    # Bachelor's
    bachelor_keywords = [
        "bachelor of technology", "bachelor of engineering",
        "b.tech", "btech", "b.e", "be ",
        "bsc", "b.sc", "bachelor of science",
        "bca", "bachelor of computer applications"
    ]
    if any(k in t for k in bachelor_keywords):
        return 2, "Bachelor's"

    # Diploma / 12th
    diploma_keywords = [
        "diploma in", "polytechnic",
        "12th", "hsc", "senior secondary"
    ]
    if any(k in t for k in diploma_keywords):
        return 1, "Diploma / 12th"

    return 0, "Unknown"


# ✅ ✅ CACHED WRAPPER USED EVERYWHERE
@lru_cache(maxsize=2048)
def analyze_resume_experience_education(text: str) -> Tuple[float, int, str]:
    """
    Returns:
        (years_experience, education_level, education_label)

    Cached so the same resume text is not processed repeatedly.
    """
    years = extract_years_experience(text)
    edu_level, edu_label = extract_education_level(text)
    return years, edu_level, edu_label
