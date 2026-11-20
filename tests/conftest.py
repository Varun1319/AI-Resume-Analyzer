import hashlib

import numpy as np
import pytest

from resume_analyzer.features import builder
from resume_analyzer.features.semantic import BertResumeMatcher


class FakeSentenceModel:
    """Deterministic bag-of-words stand-in for Sentence-BERT that records each encode call."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        self.calls.append(list(texts))
        vecs = np.zeros((len(texts), 64))
        for i, text in enumerate(texts):
            for word in text.lower().split():
                vecs[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)


@pytest.fixture
def fake_model():
    return FakeSentenceModel()


@pytest.fixture
def matcher(fake_model):
    m = BertResumeMatcher.__new__(BertResumeMatcher)  # skip loading the real model
    m.model = fake_model
    return m


@pytest.fixture(scope="session")
def skill_extractor():
    from resume_analyzer.features.skills import SkillExtractor

    try:
        return SkillExtractor()
    except OSError:
        pytest.skip("spaCy model en_core_web_sm is not installed")


@pytest.fixture
def offline_features(monkeypatch, matcher, skill_extractor):
    """Build features with the fake encoder; TF-IDF, skills and profile parsing stay real."""
    monkeypatch.setattr(builder, "get_bert_matcher", lambda: matcher)
    monkeypatch.setattr(builder, "get_skill_extractor", lambda: skill_extractor)
    return matcher.model
