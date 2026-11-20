# ui/recruiter.py: rank many resumes against one job description
import numpy as np
import pandas as pd
import streamlit as st

from resume_analyzer.ranking.explain import explain_match
from resume_analyzer.ranking.ranker import score_pairs
from resume_analyzer.ui.common import chunks_from_pdf_bytes


def render() -> None:
    st.markdown("<h4>Multi-Resume Screening & Ranking</h4>", unsafe_allow_html=True)
    st.write("Upload multiple resumes and rank them against a single job description.")

    job_desc = st.text_area(
        "Paste the job description / JD for ranking",
        placeholder="Paste the full job description text used to rank all resumes…",
        height=200,
        key="multi_jd"
    )

    uploaded_pdfs = st.file_uploader(
        "Upload multiple resume PDFs",
        type="pdf",
        accept_multiple_files=True,
        key="multi_resume_uploader"
    )

    if st.button("Rank Resumes"):
        _rank(job_desc, uploaded_pdfs)

    if "multi_rank_data" in st.session_state:
        _explain_top_candidate()


def _rank(job_desc: str, uploaded_pdfs) -> None:
    if not job_desc.strip():
        st.warning("Please paste a job description first.")
        return
    if not uploaded_pdfs:
        st.warning("Please upload at least one resume PDF.")
        return

    with st.spinner("Reading resumes and computing scores..."):
        resume_names = [pdf.name for pdf in uploaded_pdfs]
        resume_texts = [" ".join(chunks_from_pdf_bytes(pdf.getvalue())) for pdf in uploaded_pdfs]
        scores = score_pairs([(job_desc, text) for text in resume_texts])

        df_rank = pd.DataFrame({
            "Resume": resume_names,
            "Match Score": [s * 100 for s in scores]
        }).sort_values("Match Score", ascending=False).reset_index(drop=True)

        st.session_state["multi_rank_data"] = {
            "job_desc": job_desc,
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


def _explain_top_candidate() -> None:
    st.markdown("### 🔍 Explain Top Resume")

    if not st.button("Explain Top Candidate"):
        return

    data = st.session_state["multi_rank_data"]
    best_idx = int(np.argmax(data["scores"]))
    best_name = data["resume_names"][best_idx]

    try:
        df_shap, base_value = explain_match(data["job_desc"], data["resume_texts"][best_idx])
    except RuntimeError as e:
        st.error(str(e))
        return

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
