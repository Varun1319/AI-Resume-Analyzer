# linkedin_scraper.py
from __future__ import annotations

import time
from typing import List

import pandas as pd
import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)

from ml_model import LocalMLModel
from resume_analyzer import ResumeAnalyzer  # ✅ use existing pdf_to_chunks


class LinkedInScraper:
    """Simple public LinkedIn Jobs scraper for demo purposes.

    NOTE:
    - Uses only the public jobs search (no login).
    - Be gentle with job_count and usage to avoid hitting limits.
    """

    @staticmethod
    def webdriver_setup() -> webdriver.Chrome:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1440,900")
        # Slightly reduce logging noise
        options.add_experimental_option("excludeSwitches", ["enable-logging"])

        try:
            driver = webdriver.Chrome(options=options)
        except WebDriverException as e:
            st.error("Failed to start Chrome WebDriver. Make sure ChromeDriver is installed and on PATH.")
            raise e

        return driver

    @staticmethod
    def _build_search_url(job_title: str, job_location: str, country_domain: str = "in") -> str:
        from urllib.parse import quote_plus

        base = f"https://{country_domain}.linkedin.com/jobs/search"
        params = f"?keywords={quote_plus(job_title)}&location={quote_plus(job_location)}"
        return base + params

    @staticmethod
    def _scroll_until_enough_jobs(driver, max_scrolls: int, min_jobs: int) -> None:
        """Scroll the page down multiple times until we have at least `min_jobs`
        job cards or we hit `max_scrolls`.
        """
        body = driver.find_element(By.TAG_NAME, "body")
        last_len = 0

        for _ in range(max_scrolls):
            body.send_keys(Keys.END)
            time.sleep(1.5)  # gentle wait for new cards to load

            cards = driver.find_elements(By.CSS_SELECTOR, "div.base-card")
            curr_len = len(cards)
            if curr_len >= min_jobs:
                break
            # If nothing new is loading, stop early
            if curr_len == last_len:
                break
            last_len = curr_len

    @staticmethod
    def get_data(job_title: str, job_location: str, job_count: int, country_domain: str = "in") -> pd.DataFrame:
        driver = LinkedInScraper.webdriver_setup()
        url = LinkedInScraper._build_search_url(job_title, job_location, country_domain)

        try:
            driver.get(url)

            # Wait until at least one job card is visible or timeout
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.base-card"))
            )

            # Scroll until we have enough jobs or hit scroll limit
            LinkedInScraper._scroll_until_enough_jobs(
                driver,
                max_scrolls=10,          # be gentle; can tweak
                min_jobs=job_count * 2,   # overshoot a bit, we dedupe later
            )

            # Extract job info
            cards = driver.find_elements(By.CSS_SELECTOR, "div.base-card")
            companies: List[str] = []
            titles: List[str] = []
            locations: List[str] = []
            urls: List[str] = []

            for card in cards:
                try:
                    company_el = card.find_element(By.CSS_SELECTOR, "h4.base-search-card__subtitle")
                    title_el = card.find_element(By.CSS_SELECTOR, "h3.base-search-card__title")
                    loc_el = card.find_element(By.CSS_SELECTOR, "span.job-search-card__location")
                    link_el = card.find_element(By.CSS_SELECTOR, "a.base-card__full-link")
                except NoSuchElementException:
                    continue  # skip malformed cards

                companies.append(company_el.text.strip())
                titles.append(title_el.text.strip())
                locations.append(loc_el.text.strip())
                urls.append(link_el.get_attribute("href"))

            # Deduplicate by URL while preserving order
            seen = set()
            unique_rows = []
            for c, t, l, u in zip(companies, titles, locations, urls):
                if u in seen:
                    continue
                seen.add(u)
                unique_rows.append((c, t, l, u))

            df = pd.DataFrame(unique_rows, columns=["Company", "Job Title", "Location", "URL"])
            return df.head(job_count)

        finally:
            driver.quit()

    # ------------ STREAMLIT UI ------------ #

    @staticmethod
    @st.cache_data(show_spinner=False)
    def cached_jobs(
        job_title: str,
        job_location: str,
        job_count: int,
        country_domain: str = "in",
    ) -> pd.DataFrame:
        """Cached wrapper around the scraper so repeated queries are fast."""
        return LinkedInScraper.get_data(job_title, job_location, job_count, country_domain)

    @staticmethod
    def main():
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
            # Reuse existing ResumeAnalyzer.pdf_to_chunks
            chunks = ResumeAnalyzer.pdf_to_chunks(linkedin_resume)
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
                    df = LinkedInScraper.cached_jobs(
                        job_title, job_location, job_count, country_domain
                    )
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

        if df is not None and not df.empty:
            st.subheader("🔎 Job Results")

            # Show each job with an Apply button (your original format)
            for _, row in df.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])

                    with c1:
                        st.markdown(f"**{row['Job Title']}**")
                        st.write(row["Company"])
                        st.caption(row["Location"])
                        st.write(row["URL"])

                    with c2:
                        apply_url = row["URL"]
                        st.markdown(
                            f"""
                            <a href="{apply_url}" target="_blank">
                                <button style="
                                    background-color:#0A66C2;
                                    color:white;
                                    border:none;
                                    padding:8px 16px;
                                    border-radius:6px;
                                    cursor:pointer;
                                    font-weight:600;
                                ">
                                    Apply
                                </button>
                            </a>
                            """,
                            unsafe_allow_html=True,
                        )

            # Download original jobs
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "linkedin_jobs.csv",
                "text/csv",
            )

            # ---------- 4) Rank vs LinkedIn resume ----------
            st.markdown("---")
            st.subheader("🎯 Rank These Jobs Against Uploaded Resume")

            resume_text = st.session_state.get("linkedin_resume_text")
            if not resume_text:
                st.warning("Upload a resume above to enable job ranking.")
            else:
                if st.button("Compute Job Fit Ranking", key="rank_jobs"):
                    # Use Title + Company + Location as pseudo job description
                    job_texts = (
                        df["Job Title"].fillna("") + " | " +
                        df["Company"].fillna("") + " | " +
                        df["Location"].fillna("")
                    ).tolist()

                    with st.spinner("Computing match scores..."):
                        scores = LocalMLModel.compute_fit_scores_for_jobs(
                            resume_text, job_texts
                        )

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
        else:
            st.info("Scrape some jobs first to see results here.")
