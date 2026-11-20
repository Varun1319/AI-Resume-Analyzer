import pytest

from resume_analyzer.documents import chunk_text
from resume_analyzer.features.profile import extract_education_level, extract_years_experience
from resume_analyzer.features.skills import load_skill_terms


@pytest.mark.parametrize("text, years", [
    ("3 years of experience in Python", 3.0),
    ("2 yrs experience", 2.0),
    ("5+ years building APIs", 5.0),
    ("Worked 2 years at A and 4 years at B", 4.0),
    ("Fresher looking for a first role", 0.0),
    ("No numbers here", 0.0),
    ("", 0.0),
])
def test_extract_years_experience(text, years):
    assert extract_years_experience(text) == years


@pytest.mark.parametrize("text, level, label", [
    ("PhD in Physics", 4, "PhD"),
    ("Master of Science in CS", 3, "Master's"),
    ("B.Tech in Computer Science", 2, "Bachelor's"),
    ("Diploma in Electronics", 1, "Diploma / 12th"),
    ("Self-taught", 0, "Unknown"),
])
def test_extract_education_level(text, level, label):
    assert extract_education_level(text) == (level, label)


def test_skill_match_is_share_of_jd_skills(skill_extractor):
    score, resume_skills, jd_skills = skill_extractor.skill_match(
        "Built services in Python and Docker on AWS.",
        "We need Python, AWS, Kubernetes and Docker experience.",
    )
    assert jd_skills == {"python", "aws", "kubernetes", "docker"}
    assert resume_skills == {"python", "docker", "aws"}
    assert score == 0.75


def test_multi_word_skills_are_matched(skill_extractor):
    assert {"machine learning", "spring boot"} <= skill_extractor.extract_skills("Machine learning with Spring Boot")


def test_no_jd_skills_scores_zero(skill_extractor):
    assert skill_extractor.skill_match("python", "We value curiosity")[0] == 0.0


def test_skill_file_ignores_comments_and_blank_lines(tmp_path):
    path = tmp_path / "skills.txt"
    path.write_text("# comment\nPython\n\n  machine learning  \n")
    assert load_skill_terms(path) == {"python", "machine learning"}


def test_chunks_overlap_and_cover_the_text():
    text = "".join(chr(97 + i % 26) for i in range(1500))
    chunks = chunk_text(text, chunk_size=700, overlap=200)
    assert chunks[0] == text[:700]
    assert chunks[1] == text[500:1200]
    assert chunks[1][:200] == chunks[0][-200:]
    assert text.endswith(chunks[-1])


def test_short_and_empty_text():
    assert chunk_text("hello", 700, 200) == ["hello"]
    assert chunk_text("", 700, 200) == []
    assert chunk_text("   \n\n  ", 700, 200) == []


def test_pair_scores_encode_each_distinct_text_once(matcher, fake_model):
    scores = matcher.pair_scores([("jd", "resume a"), ("jd", "resume b"), ("jd", "resume a")])
    assert len(fake_model.calls) == 1
    assert sorted(fake_model.calls[0]) == ["jd", "resume a", "resume b"]
    assert scores[0] == scores[2]


def test_pair_scores_are_cosine_similarity(matcher):
    same, unrelated = matcher.pair_scores([
        ("python developer", "python developer"),
        ("python developer", "sales manager"),
    ])
    assert same == pytest.approx(1.0)
    assert unrelated < same
