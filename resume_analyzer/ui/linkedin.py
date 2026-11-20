# ui/linkedin.py: LinkedIn job search, ranked against an uploaded resume
import pandas as pd
import streamlit as st

from resume_analyzer.jobs.linkedin import get_jobs
from resume_analyzer.ranking.ranker import score_pairs
from resume_analyzer.ui.common import chunks_from_pdf_bytes


@st.cache_data(show_spinner=False)
def _cached_jobs(job_title: str, job_location: str, job_count: int, country_domain: str = "in") -> pd.DataFrame:
    """Cached wrapper around the scraper so repeated queries are fast."""
    return get_jobs(job_title, job_location, job_count, country_domain)


def render() -> None:
    st.subheader("💼 LinkedIn Job Scraper (Public Search)")
    st.caption(
        "Fetches jobs from the public LinkedIn jobs page. "
        "Results may vary based on LinkedIn UI and rate limits."
    )

    # ---------- 0) Resume upload just for LinkedIn ranking ----------
    st.markdown("### 📄 Upload Resume for Job Matching")
    linkedin_resume = st.file_uploader(
        "Upload Resume (used only for job ranking)",
        type="pdf",
        key="linkedin_resume_uploader",
    )

    if linkedin_resume is not None:
        chunks = chunks_from_pdf_bytes(linkedin_resume.getvalue())
        st.session_state["linkedin_resume_text"] = " ".join(chunks)
        st.success("✅ Resume loaded for LinkedIn job ranking")

    # ---------- 1) Job search inputs ----------
    col1, col2 = st.columns(2)
    with col1:
        job_title = st.text_input("Job Title", "Data Scientist")
    with col2:
        job_location = st.text_input("Location", "India")

    job_count = st.number_input("Number of jobs", 1, 50, 10)

    # optional: choose LinkedIn country domain
    country_domain = st.selectbox(
        "LinkedIn domain",
        options=["in", "www", "de", "fr", "uk"],
        index=0,
        format_func=lambda x: f"{x}.linkedin.com",
    )

    # ---------- 2) Scrape + store in session ----------
    if st.button("Scrape Jobs", key="scrape_jobs"):
        with st.spinner("Scraping LinkedIn jobs..."):
            try:
                df = _cached_jobs(job_title, job_location, job_count, country_domain)
            except Exception as e:
                st.error(f"Scraping failed: {e}")
                return

        if df.empty:
            st.warning("No jobs found. Try changing the title, location, or reducing job count.")
            return

        # store in session so next run we still have the data
        st.session_state["linkedin_jobs_df"] = df
        st.success(f"Found {len(df)} jobs.")

    # ---------- 3) Read df from session and show results ----------
    df = st.session_state.get("linkedin_jobs_df")
    if df is None or df.empty:
        st.info("Scrape some jobs first to see results here.")
        return

    _show_jobs(df)
    _rank_jobs(df)


def _show_jobs(df: pd.DataFrame) -> None:
    st.subheader("🔎 Job Results")

    for _, row in df.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])

            with c1:
                st.markdown(f"**{row['Job Title']}**")
                st.write(row["Company"])
                st.caption(row["Location"])
                st.write(row["URL"])

            with c2:
                # The URL is scraped from an external page, so it is never
                # rendered as raw HTML, and only web links get a button.
                if str(row["URL"]).startswith(("https://", "http://")):
                    st.link_button("Apply", row["URL"])

    # Download original jobs
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "linkedin_jobs.csv",
        "text/csv",
    )


def _rank_jobs(df: pd.DataFrame) -> None:
    st.markdown("---")
    st.subheader("🎯 Rank These Jobs Against Uploaded Resume")

    resume_text = st.session_state.get("linkedin_resume_text")
    if not resume_text:
        st.warning("Upload a resume above to enable job ranking.")
        return

    if not st.button("Compute Job Fit Ranking", key="rank_jobs"):
        return

    # Use Title + Company + Location as pseudo job description
    job_texts = (
        df["Job Title"].fillna("") + " | " +
        df["Company"].fillna("") + " | " +
        df["Location"].fillna("")
    ).tolist()

    with st.spinner("Computing match scores..."):
        scores = score_pairs([(job_text, resume_text) for job_text in job_texts])

    df_rank = df.copy()
    df_rank["Match Score (%)"] = [round(s * 100, 2) for s in scores]
    df_rank = df_rank.sort_values("Match Score (%)", ascending=False).reset_index(drop=True)

    st.success("✅ Jobs ranked against your uploaded resume")
    st.dataframe(
        df_rank[["Company", "Job Title", "Location", "Match Score (%)", "URL"]],
        use_container_width=True,
    )

    st.download_button(
        "Download Ranked Jobs CSV",
        df_rank.to_csv(index=False).encode("utf-8"),
        "linkedin_jobs_ranked.csv",
        "text/csv",
    )
