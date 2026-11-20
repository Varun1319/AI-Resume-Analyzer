import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from resume_analyzer.features.builder import FEATURE_COLUMNS, build_feature_rows
from resume_analyzer.ranking import explain, ranker

JD = "Python developer with AWS and Docker. 3 years of experience. Bachelor of Engineering."
RESUMES = [
    "Python developer, 4 years of experience with AWS and Docker. B.Tech in computer science.",
    "Sales manager with 6 yrs experience in retail.",
    "Machine learning engineer: python, docker, kubernetes. Master of Science.",
]


@pytest.fixture
def tiny_model():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (200, len(FEATURE_COLUMNS)))
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    return GradientBoostingClassifier(n_estimators=20, random_state=0).fit(X, y)


@pytest.fixture
def no_trained_model(monkeypatch):
    monkeypatch.setattr(ranker, "load_ranker", lambda: None)


@pytest.fixture
def fresh_explainer():
    explain._get_explainer.cache_clear()
    yield
    explain._get_explainer.cache_clear()


def test_heuristic_score_weights():
    feats = {"tfidf_score": 0.5, "bert_score": 1.0, "skill_score": 0.5, "exp_score": 1.0, "edu_score": 0.6}
    text = 0.4 * 0.5 + 0.6 * 1.0
    assert ranker.heuristic_score(feats) == pytest.approx(0.6 * text + 0.2 * 0.5 + 0.1 * 1.0 + 0.1 * 0.6)


def test_batched_features_match_one_at_a_time(offline_features):
    pairs = [(JD, r) for r in RESUMES]
    batched = build_feature_rows(pairs)
    assert len(offline_features.calls) == 1

    one_by_one = [build_feature_rows([p])[0] for p in pairs]
    for row, single in zip(batched, one_by_one):
        assert list(row) == FEATURE_COLUMNS
        assert row == pytest.approx(single)


def test_blank_pairs_score_zero_without_computing_features(offline_features, no_trained_model):
    scores = ranker.score_pairs([(JD, RESUMES[0]), (JD, "   "), ("", RESUMES[1])])
    assert scores[0] > 0 and scores[1:] == [0.0, 0.0]
    assert offline_features.calls == [[JD, RESUMES[0]]]


def test_closer_resume_ranks_higher(offline_features, no_trained_model):
    scores = ranker.score_pairs([(JD, r) for r in RESUMES])
    assert scores[0] == max(scores)
    assert scores[1] == min(scores)


def test_trained_ranker_scores_all_rows_in_one_prediction(offline_features, tiny_model, monkeypatch):
    calls = []
    predict = tiny_model.predict_proba
    monkeypatch.setattr(tiny_model, "predict_proba", lambda X: calls.append(len(X)) or predict(X))
    monkeypatch.setattr(ranker, "load_ranker", lambda: tiny_model)

    pairs = [(JD, r) for r in RESUMES]
    scores = ranker.score_pairs(pairs)

    X = np.array([[row[c] for c in FEATURE_COLUMNS] for row in build_feature_rows(pairs)])
    assert calls == [3]
    assert scores == pytest.approx(predict(X)[:, 1].tolist())


def test_falls_back_to_heuristic_when_ranker_fails(offline_features, monkeypatch):
    class Broken:
        def predict_proba(self, X):
            raise ValueError("feature mismatch")

    monkeypatch.setattr(ranker, "load_ranker", lambda: Broken())
    pairs = [(JD, RESUMES[0])]
    assert ranker.score_pairs(pairs) == [pytest.approx(ranker.heuristic_score(build_feature_rows(pairs)[0]))]


def test_shap_values_add_up_to_the_model_output(offline_features, tiny_model, monkeypatch, fresh_explainer):
    pytest.importorskip("shap")
    monkeypatch.setattr(explain, "load_ranker", lambda: tiny_model)

    df, base = explain.explain_match(JD, RESUMES[0])

    row = build_feature_rows([(JD, RESUMES[0])])[0]
    x = np.array([[row[c] for c in FEATURE_COLUMNS]])
    assert set(df["feature"]) == set(FEATURE_COLUMNS)
    assert base + df["shap_value"].sum() == pytest.approx(tiny_model.decision_function(x)[0])


def test_explain_without_trained_model_raises(monkeypatch, fresh_explainer):
    monkeypatch.setattr(explain, "load_ranker", lambda: None)
    with pytest.raises(RuntimeError, match="Train the ranker first"):
        explain.explain_match(JD, RESUMES[0])
