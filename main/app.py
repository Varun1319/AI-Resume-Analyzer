import hashlib
from typing import List
import numpy as np
from shap_explainer import explain_single
from skills_extractor import skill_extractor
from experience_education import analyze_resume_experience_education
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from ui_utils import streamlit_config
from resume_analyzer import ResumeAnalyzer
from ml_model import LocalMLModel
from linkedin_scraper import LinkedInScraper
import os

@st.cache_data(show_spinner=False)
def cached_pdf_to_chunks(file_bytes: bytes) -> List[str]:
    from io import BytesIO
    fake_file = BytesIO(file_bytes)
    return ResumeAnalyzer.pdf_to_chunks(fake_file)


# ✅ must be first Streamlit call
st.set_page_config(page_title="Resume Analyzer AI", layout="wide")

# ---------------------- MAIN APP ---------------------- #
def main():
    streamlit_config()
    add_vertical_space(2)
    with st.sidebar:
        st.markdown("### Mode")
        mode = st.radio(
            "Select mode",
            options=["👤 Candidate", "🧑‍💼 Recruiter"],
            index=0,
            key="app_mode",
            label_visibility="collapsed",  # keeps Streamlit happy, label hidden
        )

        st.markdown("---")

        if "Candidate" in mode:
            option = option_menu(
                menu_title="Candidate Mode",
                options=[
                    "Summary",
                    "Strength",
                    "Weakness",
                    "Job Titles",
                    "Fit Score",
                    "LinkedIn Jobs",  # 👈 moved here
                ],
                icons=[
                    "file-text",
                    "bar-chart",
                    "x-circle",
                    "list-ul",
                    "speedometer",
                    "linkedin",
                ],
            )
        else:
            option = option_menu(
                menu_title="Recruiter Mode",
                options=[
                    "Multi-Resume Ranking",
                ],
                icons=[
                    "layers",
                ],
            )


    # Read current mode from session (sidebar sets this later via key="app_mode")
    mode = st.session_state.get("app_mode", "👤 Candidate")

    # ---- MODE INFO BANNER ----
       # ---- MODE INFO BANNER ----
    if "Candidate" in mode:
        st.info("👤 Candidate Mode – Analyze and improve a single resume.")
    else:
        st.info("🧑‍💼 Recruiter Mode – Rank and inspect multiple candidates.")

    # ================= CANDIDATE MODE: API KEY + RESUME UPLOAD =================
    uploaded_pdf = None

    # ---- API KEY (only for LLM tabs in Candidate mode) ----
    if "Candidate" in mode and option in ["Summary", "Strength", "Weakness", "Job Titles"]:
        st.markdown("### 🔑 OpenAI API Key")

        # Init session value if missing
        if "openai_api_key" not in st.session_state:
            st.session_state["openai_api_key"] = ""

        # Single text input, unique key
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state["openai_api_key"],
            key="candidate_api_key_input",
            help="Paste your OpenAI API key here (starts with 'sk-').",
        )

        # Keep in a stable place for ResumeAnalyzer._make_client
        st.session_state["openai_api_key"] = api_key_input

    # ---- PDF UPLOAD FOR SINGLE RESUME (all Candidate tabs EXCEPT LinkedIn) ----
    if "Candidate" in mode and option in ["Summary", "Strength", "Weakness", "Job Titles", "Fit Score"]:
        st.markdown("### 📄 Upload Your Resume")
        uploaded_pdf = st.file_uploader(
            "Upload Resume PDF",
            type="pdf",
            key="resume_uploader",
        )

        if uploaded_pdf is not None:
            file_bytes = uploaded_pdf.getvalue()
            chunks = cached_pdf_to_chunks(file_bytes)
            st.session_state["resume_chunks"] = chunks

            # Only run heavy OpenAI analysis on LLM tabs
            if option in ["Summary", "Strength", "Weakness", "Job Titles"]:
                ResumeAnalyzer.run_all(chunks)


    # 👉 After this, keep your sidebar code and page logic as you already have:
    # with st.sidebar: ...
    #   mode = st.radio(... key="app_mode" ...)
    #   option = option_menu(...)
    # then your big if/elif on `option`


    # ---- SIDEBAR MENU ----
        # ---- SIDEBAR: MODE + MENU ----
        # ---- SIDEBAR: MODE + MENU ----
    




    # ---- PAGE LOGIC ----
    if option in ["Summary", "Strength", "Weakness", "Job Titles"] and "resume_chunks" not in st.session_state:
        st.warning("Please upload your resume first.")

    elif option == "Summary":
        st.markdown("<h4>Summary</h4>", unsafe_allow_html=True)
        st.write(st.session_state.get("resume_summary", "⚙️ Waiting for analysis..."))

    elif option == "Strength":
        st.markdown("<h4>Strengths</h4>", unsafe_allow_html=True)
        st.write(st.session_state.get("resume_strength", "⚙️ Waiting for analysis..."))

    elif option == "Weakness":
        st.markdown("<h4>Weaknesses & Suggestions</h4>", unsafe_allow_html=True)
        st.write(st.session_state.get("resume_weakness", "⚙️ Waiting for analysis..."))

    elif option == "Job Titles":
        st.markdown("<h4>Suggested Job Titles</h4>", unsafe_allow_html=True)
        st.write(st.session_state.get("resume_jobtitles", "⚙️ Waiting for analysis..."))

    elif option == "Fit Score":
        if "resume_chunks" not in st.session_state:
            st.warning("Please upload your resume first.")
        else:
            st.markdown("<h4>AI Match Score vs Job Description</h4>", unsafe_allow_html=True)

            job_desc = st.text_area(
                "Paste the job description / JD here",
                placeholder="Paste the full job description text…"
            )

            if st.button("Compute Match Score"):
                full_resume_text = " ".join(st.session_state["resume_chunks"])

                # ---- Core hybrid score (TF-IDF + BERT + skills + exp + edu) ----
                score = LocalMLModel.compute_fit_score(job_desc, full_resume_text)

                # ---- Skills breakdown ----
                skill_score, resume_skills, jd_skills = skill_extractor.skill_match(
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
                if score >= 0.7:
                    st.success("This resume is a strong match for this job.")
                elif score >= 0.4:
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



    elif option == "Multi-Resume Ranking":
        st.markdown("<h4>Multi-Resume Screening & Ranking</h4>", unsafe_allow_html=True)
        st.write("Upload multiple resumes and rank them against a single job description.")

        job_desc_multi = st.text_area(
            "Paste the job description / JD for ranking",
            placeholder="Paste the full job description text used to rank all resumes…",
            height=200,
            key="multi_jd"
        )

        uploaded_pdfs_multi = st.file_uploader(
            "Upload multiple resume PDFs",
            type="pdf",
            accept_multiple_files=True,
            key="multi_resume_uploader"
        )

        if st.button("Rank Resumes"):
            if not job_desc_multi.strip():
                st.warning("Please paste a job description first.")
            elif not uploaded_pdfs_multi:
                st.warning("Please upload at least one resume PDF.")
            else:
                resume_names = []
                resume_texts = []

                with st.spinner("Reading resumes and computing scores..."):
                    for pdf in uploaded_pdfs_multi:
                        file_bytes = pdf.getvalue()
                        chunks = cached_pdf_to_chunks(file_bytes)
                        full_text = " ".join(chunks)
                        resume_names.append(pdf.name)
                        resume_texts.append(full_text)

                    scores = LocalMLModel.compute_fit_scores(job_desc_multi, resume_texts)

                    df_rank = pd.DataFrame({
                        "Resume": resume_names,
                        "Match Score": [s * 100 for s in scores]
                    }).sort_values("Match Score", ascending=False).reset_index(drop=True)

                    st.session_state["multi_rank_data"] = {
                    "job_desc": job_desc_multi,
                    "resume_names": resume_names,
                    "resume_texts": resume_texts,
                    "scores": scores,
                }

                st.subheader("Ranked Resumes")
                st.dataframe(df_rank)

                st.download_button(
                    "Download Ranking as CSV",
                    data=df_rank.to_csv(index=False).encode("utf-8"),
                    file_name="multi_resume_ranking.csv",
                    mime="text/csv"
                )
                # ---- Explain Top Resume (SHAP on best candidate) ----
        if "multi_rank_data" in st.session_state:
            st.markdown("### 🔍 Explain Top Resume")

            if st.button("Explain Top Candidate"):
                data = st.session_state["multi_rank_data"]
                jd_text = data["job_desc"]
                resume_texts = data["resume_texts"]
                scores = np.array(data["scores"])

                # index of best-scoring resume
                best_idx = int(np.argmax(scores))
                best_resume_text = resume_texts[best_idx]
                best_name = data["resume_names"][best_idx]

                try:
                    df_shap, base_value = explain_single(jd_text, best_resume_text)
                except RuntimeError as e:
                    st.error(str(e))
                else:
                    st.info(f"Explaining SHAP contributions for **{best_name}** (top-ranked resume).")

                    st.markdown("#### Feature Contributions")
                    st.write(
                        "Positive SHAP values push the score **higher** (better match); "
                        "negative values push it **lower**."
                    )

                    st.dataframe(
                        df_shap[["feature", "value", "shap_value", "abs_shap"]],
                        use_container_width=True,
                    )

                    st.markdown("#### ℹ️ Notes")
                    st.write(
                        f"- **Base value (model bias)**: {base_value:.3f} (average prediction)\n"
                        "- Features at the top had the **largest impact** on this candidate's score."
                    )
        
    

    elif option == "Explainability":
        st.markdown("<h4>Model Explainability (SHAP)</h4>", unsafe_allow_html=True)
        st.write(
            "See which features (text similarity, skills, experience, education) "
            "influenced the model's match score for a given JD–resume pair."
        )

        # JD input
        jd_text = st.text_area(
            "Paste the job description / JD here",
            placeholder="Paste the same JD you used for Fit Score / Ranking…",
            height=200,
            key="explain_jd",
        )

        # Resume text: reuse uploaded resume if available
        if "resume_chunks" in st.session_state:
            st.info("Using the currently uploaded resume for explanation.")
            resume_text = " ".join(st.session_state["resume_chunks"])
        else:
            st.warning("No resume uploaded yet. Please upload a resume on the main page first.")
            resume_text = None

        if st.button("Explain this prediction"):
            if not jd_text.strip():
                st.warning("Please paste a job description first.")
            elif not resume_text:
                st.warning("Please upload a resume first.")
            else:
                try:
                    df_shap, base_value = explain_single(jd_text, resume_text)
                except RuntimeError as e:
                    st.error(str(e))
                else:
                    st.markdown("### 🔍 Feature Contributions")
                    st.write(
                        "Positive SHAP values push the score **higher** (towards a better match), "
                        "negative values push it **lower**."
                    )

                    st.dataframe(
                        df_shap[["feature", "value", "shap_value", "abs_shap"]],
                        use_container_width=True,
                    )

                    st.markdown("### ℹ️ Notes")
                    st.write(
                        f"- **Base value (model bias)**: {base_value:.3f} (average prediction)\n"
                        "- Features at the top of the table had the **largest impact** "
                        "on this candidate's score for this JD."
                    )

    elif option == "LinkedIn Jobs":
        LinkedInScraper.main()
        


if __name__ == "__main__":
    main()
