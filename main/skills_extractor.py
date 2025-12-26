# skills_extractor.py

from __future__ import annotations
from typing import Set
from functools import lru_cache
import spacy


class SkillExtractor:
    """
    Extracts technical skills from free-form resume / JD text.
    Uses:
    - spaCy for tokenization
    - A curated dictionary of skill terms
    """

    def __init__(self) -> None:
        # Load spaCy model with unnecessary components disabled
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])

        # Canonical skill list (you can expand this anytime)
        self.skill_terms: Set[str] = {
            "python", "java", "c", "c++", "c#", "javascript", "typescript",
            "react", "angular", "vue",
            "node", "nodejs", "node.js",
            "django", "flask", "fastapi",
            "spring", "spring boot",
            "html", "css", "bootstrap", "tailwind",
            "sql", "mysql", "postgres", "mongodb", "redis",
            "aws", "azure", "gcp",
            "docker", "kubernetes",
            "git", "github",
            "linux", "unix",
            "machine learning", "deep learning", "nlp",
            "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
            "rest", "rest api", "graphql",
        }

    # -------------------- INTERNAL CACHED CORE -------------------- #
    @lru_cache(maxsize=2048)
    def _extract_skills_cached(self, text_low: str) -> Set[str]:
        """
        Cached core function. This is where spaCy actually runs.
        text_low MUST be lowercase.
        """
        doc = self.nlp(text_low)
        found: Set[str] = set()

        # 1) Unigram matches
        for token in doc:
            t = token.text.strip()
            if t in self.skill_terms:
                found.add(t)

        # 2) Bigram / trigram phrase matches
        tokens = [t.text for t in doc]

        for i in range(len(tokens)):
            if i + 1 < len(tokens):
                bigram = f"{tokens[i]} {tokens[i + 1]}"
                if bigram in self.skill_terms:
                    found.add(bigram)

            if i + 2 < len(tokens):
                trigram = f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}"
                if trigram in self.skill_terms:
                    found.add(trigram)

        return found

    # -------------------- PUBLIC API -------------------- #
    def extract_skills(self, text: str) -> Set[str]:
        """
        Public skill extractor.
        Normalizes text and calls the cached core.
        """
        if not text:
            return set()
        text_low = text.lower()
        return self._extract_skills_cached(text_low)

    def skill_match(self, resume_text: str, jd_text: str):
        """
        Computes:
        - Resume skill set
        - JD skill set
        - Overlap-based normalized match score ∈ [0,1]
        """
        resume_skills = self.extract_skills(resume_text)
        jd_skills = self.extract_skills(jd_text)

        if not resume_skills or not jd_skills:
            return 0.0, resume_skills, jd_skills

        common = resume_skills.intersection(jd_skills)
        score = len(common) / max(len(jd_skills), 1)

        return float(score), resume_skills, jd_skills


# ✅ SINGLETON INSTANCE USED THROUGHOUT THE APP
skill_extractor = SkillExtractor()
