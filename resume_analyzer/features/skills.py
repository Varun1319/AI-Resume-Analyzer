# features/skills.py

from __future__ import annotations
from pathlib import Path
from typing import Set
from functools import lru_cache
import spacy

from resume_analyzer.config import SKILLS_PATH


def load_skill_terms(path: Path = SKILLS_PATH) -> Set[str]:
    """Skill dictionary: one skill per line; blank lines and '#' comments are ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {line.strip().lower() for line in lines if line.strip() and not line.lstrip().startswith("#")}


class SkillExtractor:
    """
    Extracts technical skills from free-form resume / JD text.
    Uses:
    - spaCy for tokenization
    - A curated dictionary of skill terms (data/skills.txt)
    """

    def __init__(self) -> None:
        # Load spaCy model with unnecessary components disabled
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
        self.skill_terms: Set[str] = load_skill_terms()

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


@lru_cache(maxsize=1)
def get_skill_extractor() -> SkillExtractor:
    """Shared instance; spaCy loads on first use rather than at import."""
    return SkillExtractor()
