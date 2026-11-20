# ui/candidate.py: feedback and fit score for a single resume
import hashlib

import streamlit as st
from openai import OpenAI

from resume_analyzer.config import MODERATE_MATCH_THRESHOLD, STRONG_MATCH_THRESHOLD
from resume_analyzer.features.profile import analyze_resume_experience_education
from resume_analyzer.features.skills import get_skill_extractor
from resume_analyzer.llm.rag import analyze_resume
from resume_analyzer.ranking.ranker import score_pair
from resume_analyzer.ui.common import chunks_from_pdf_bytes

# Sidebar option -> (page heading, key in the analysis results)
FEEDBACK_PAGES = {
    "Summary": ("Summary", "summary"),
    "Strength": ("Strengths", "strengths"),
    "Weakness": ("Weaknesses & Suggestions", "weaknesses"),
    "Job Titles": ("Suggested Job Titles", "job_titles"),
}
PAGES = [*FEEDBACK_PAGES, "Fit Score"]


def render(option: str) -> None:
    needs_llm = option in FEEDBACK_PAGES
    if needs_llm:
        _api_key_input()
    _resume_upload(analyze=needs_llm)

    if needs_llm:
        _feedback_page(option)
    else:
        _fit_score_page()


def _api_key_input() -> None:
    st.markdown("### 🔑 OpenAI API Key")

    # Init session value if missing
    if "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = ""

    st.session_state["openai_api_key"] = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state["openai_api_key"],
        key="candidate_api_key_input",
        help="Paste your OpenAI API key here (starts with 'sk-').",
    )


def _resume_upload(analyze: bool) -> None:
    st.markdown("### 📄 Upload Your Resume")
    uploaded_pdf = st.file_uploader("Upload Resume PDF", type="pdf", key="resume_uploader")
    if uploaded_pdf is None:
        return

    file_bytes = uploaded_pdf.getvalue()
    chunks = chunks_from_pdf_bytes(file_bytes)
    st.session_state["resume_chunks"] = chunks

    # Streamlit reruns this script on every interaction, so analyze each upload only once.
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    if not analyze or st.session_state.get("analyzed_resume_hash") == file_hash:
        return

    api_key = st.session_state.get("openai_api_key")
    if not api_key:
        st.warning("⚠️ Please paste your OpenAI API key in the box above.")
        st.stop()

    st.info("⏳ Generating all insights (summary, strengths, weaknesses, titles)...")
    st.session_state["resume_analysis"] = analyze_resume(OpenAI(api_key=api_key), chunks)
    st.session_state["analyzed_resume_hash"] = file_hash
    st.success("✅ All resume insights generated! You can navigate using the sidebar.")


def _feedback_page(option: str) -> None:
    if "resume_chunks" not in st.session_state:
        st.warning("Please upload your resume first.")
        return

    heading, key = FEEDBACK_PAGES[option]
    st.markdown(f"<h4>{heading}</h4>", unsafe_allow_html=True)
    st.write(st.session_state.get("resume_analysis", {}).get(key, "⚙️ Waiting for analysis..."))


def _fit_score_page() -> None:
    if "resume_chunks" not in st.session_state:
        st.warning("Please upload your resume first.")
        return

    st.markdown("<h4>AI Match Score vs Job Description</h4>", unsafe_allow_html=True)

    job_desc = st.text_area(
        "Paste the job description / JD here",
        placeholder="Paste the full job description text…"
    )

    if not st.button("Compute Match Score"):
        return

    full_resume_text = " ".join(st.session_state["resume_chunks"])

    # ---- Core hybrid score (TF-IDF + BERT + skills + exp + edu) ----
    score = score_pair(job_desc, full_resume_text)

    # ---- Skills breakdown ----
    skill_score, resume_skills, jd_skills = get_skill_extractor().skill_match(
        full_resume_text, job_desc
    )
    common_skills = resume_skills.intersection(jd_skills)

    # ---- Experience & Education analysis ----
    years_exp, edu_level, edu_label = analyze_resume_experience_education(full_resume_text)

    # ---- Top-level metrics ----
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Match Score", f"{score * 100:.1f}%")
    col2.metric("Skill Match", f"{skill_score * 100:.1f}%")
    col3.metric("Experience (years)", f"{years_exp:.1f}")

    # Interpretation
    if score >= STRONG_MATCH_THRESHOLD:
        st.success("This resume is a strong match for this job.")
    elif score >= MODERATE_MATCH_THRESHOLD:
        st.info("This resume is a moderate match. You may want to tweak keywords and projects.")
    else:
        st.warning("Low match. Consider tailoring the resume more to this job's requirements.")

    # ---- Education info ----
    st.markdown("### 🎓 Education Assessment")
    st.write(f"Detected highest education level: **{edu_label}**")

    # ---- Skill details ----
    st.markdown("### 🔧 Skill Analysis")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Skills in Job Description**")
        if jd_skills:
            st.write(", ".join(sorted(jd_skills)))
        else:
            st.write("_No recognizable skills found in JD text._")

    with c2:
        st.markdown("**Skills in Your Resume**")
        if resume_skills:
            st.write(", ".join(sorted(resume_skills)))
        else:
            st.write("_No recognizable skills found in resume text._")

    with c3:
        st.markdown("**Matched Skills**")
        if common_skills:
            st.write(", ".join(sorted(common_skills)))
        else:
            st.write("_No overlapping skills found. Consider aligning your skills with the JD more clearly._")
