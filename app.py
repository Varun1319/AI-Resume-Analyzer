import time
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import warnings
warnings.filterwarnings("ignore")

# ✅ Must be first
st.set_page_config(page_title="Resume Analyzer AI", layout="wide")

# ---------------------- HEADER ---------------------- #
def streamlit_config():
    st.markdown(
        """
        <style>
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }
        h4 { color: orange; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<h1 style='text-align:center;'>Resume Analyzer AI</h1>", unsafe_allow_html=True)


# ---------------------- RESUME ANALYZER ---------------------- #
class ResumeAnalyzer:

    def pdf_to_chunks(pdf):
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n\n"
        chunk_size = 700
        overlap = 200
        chunks, start = [], 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end].strip())
            start = max(0, end - overlap)
        return [c for c in chunks if c]

    @staticmethod
    def _make_client():
        if "openai_api_key" not in st.session_state or not st.session_state.openai_api_key:
            st.warning("⚠️ Please enter your OpenAI API key above.")
            st.stop()
        return OpenAI(api_key=st.session_state.openai_api_key)

    @staticmethod
    def _get_embeddings(client, texts, model="text-embedding-3-small"):
        if not texts:
            return np.zeros((0, 0))
        resp = client.embeddings.create(model=model, input=texts)
        return [np.array(item.embedding, dtype=float) for item in resp.data]

    @staticmethod
    def _cosine_sim_matrix(matrix, vector):
        if matrix.size == 0 or vector.size == 0:
            return np.array([])
        mat_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
        vec_norm = vector / (np.linalg.norm(vector) + 1e-12)
        return np.dot(mat_norm, vec_norm)

    def openai_analyze(chunks, prompt, top_k=3):
        client = ResumeAnalyzer._make_client()
        chunk_vectors = np.vstack(ResumeAnalyzer._get_embeddings(client, chunks))
        query_vec = ResumeAnalyzer._get_embeddings(client, [prompt])[0]
        sims = ResumeAnalyzer._cosine_sim_matrix(chunk_vectors, query_vec)
        top_idx = np.argsort(-sims)[:top_k]
        selected = [chunks[i] for i in top_idx]
        context = "\n\n---\n\n".join(selected)
        full_prompt = f"""
You are an assistant that analyzes resumes carefully.

Context:
\"\"\"{context}\"\"\"

Instruction:
{prompt}

Answer clearly and professionally.
"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.5,
            max_tokens=1500
        )
        return resp.choices[0].message.content

    def summary_prompt(chunks): return f"Provide a **comprehensive and well-structured summary** of this resume. Include key skills, educational highlights, technical expertise, and achievements.Also add a concluding note on overall candidate suitability.Resume content:\n\n{chunks}"
    def strength_prompt(chunks): return f"Analyze this resume and write a **detailed list of the candidate strengths**. Include technical strengths, soft skills, leadership abilities, teamwork, achievements, and any unique differentiating factors. Give **short examples** from the text wherever possible.Resume content:\n\n{chunks}"
    def weakness_prompt(chunks): return f"Critically analyze this resume to identify **weak areas or gaps** in terms of formatting, missing keywords, or lack of quantifiable results. Suggest **specific improvements** to make it more impactful for recruiters or ATS.Resume content:\n\n{chunks}"
    def job_title_prompt(chunks): return f"Based on this resume, list **5–10 most suitable job roles or titles**, categorized by domain (e.g., Software Engineering, Data, Product, etc.) and include **a one-line reason** why each fits the profile.Resume content:\n\n{chunks}"

    @staticmethod
    def run_all(chunks):
        st.info("⏳ Generating all insights (summary, strengths, weaknesses, titles)...")
        summary = ResumeAnalyzer.openai_analyze(chunks, ResumeAnalyzer.summary_prompt(" ".join(chunks[:6])))
        strength = ResumeAnalyzer.openai_analyze(chunks, ResumeAnalyzer.strength_prompt(" ".join(chunks[:6])))
        weakness = ResumeAnalyzer.openai_analyze(chunks, ResumeAnalyzer.weakness_prompt(" ".join(chunks[:6])))
        job_titles = ResumeAnalyzer.openai_analyze(chunks, ResumeAnalyzer.job_title_prompt(" ".join(chunks[:6])))
        st.session_state["resume_summary"] = summary
        st.session_state["resume_strength"] = strength
        st.session_state["resume_weakness"] = weakness
        st.session_state["resume_jobtitles"] = job_titles
        st.success("✅ All resume insights generated! You can navigate using the sidebar.")


# ---------------------- LINKEDIN SCRAPER ---------------------- #
class LinkedInScraper:

    def webdriver_setup():
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=options)
        driver.maximize_window()
        return driver

    def get_data(driver, job_title, job_location, job_count):
        driver.get(f"https://in.linkedin.com/jobs/search?keywords={job_title}&location={job_location}")
        driver.implicitly_wait(5)
        time.sleep(3)
        for _ in range(5):
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
            time.sleep(1)
        company = [i.text for i in driver.find_elements(By.CSS_SELECTOR, "h4.base-search-card__subtitle")]
        title = [i.text for i in driver.find_elements(By.CSS_SELECTOR, "h3.base-search-card__title")]
        location = [i.text for i in driver.find_elements(By.CSS_SELECTOR, "span.job-search-card__location")]
        url = [i.get_attribute("href") for i in driver.find_elements(By.XPATH, '//a[contains(@href, "/jobs/")]')]
        min_len = min(len(company), len(title), len(location), len(url))
        df = pd.DataFrame({
            "Company": company[:min_len],
            "Job Title": title[:min_len],
            "Location": location[:min_len],
            "URL": url[:min_len]
        })
        return df.head(job_count)

    def main():
        st.subheader("💼 LinkedIn Job Scraper")
        job_title = st.text_input("Job Title", "Data Scientist")
        job_location = st.text_input("Location", "India")
        job_count = st.number_input("Job Count", 1, 30, 5)
        if st.button("Scrape Jobs"):
            with st.spinner("Scraping LinkedIn Jobs..."):
                driver = LinkedInScraper.webdriver_setup()
                df = LinkedInScraper.get_data(driver, job_title, job_location, job_count)
                driver.quit()
            st.dataframe(df)
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "linkedin_jobs.csv", "text/csv")


# ---------------------- MAIN APP ---------------------- #
if __name__ == "__main__":
    streamlit_config()
    add_vertical_space(2)

    # ---- API KEY ----
    st.markdown("### 🔑 Enter your OpenAI API Key")
    st.session_state.openai_api_key = st.text_input(
        "OpenAI API Key", 
        type="password", 
        value=st.session_state.get("openai_api_key", ""), 
        key="global_api_key"
    )

    # ---- PDF UPLOAD ----
    st.markdown("### 📄 Upload Your Resume ")
    uploaded_pdf = st.file_uploader("Upload Resume PDF", type="pdf", key="resume_uploader")

    if uploaded_pdf and "resume_chunks" not in st.session_state:
        chunks = ResumeAnalyzer.pdf_to_chunks(uploaded_pdf)
        st.session_state["resume_chunks"] = chunks
        ResumeAnalyzer.run_all(chunks)

    # ---- SIDEBAR MENU ----
    with st.sidebar:
        option = option_menu(
            menu_title="",
            options=["Summary", "Strength", "Weakness", "Job Titles", "LinkedIn Jobs"],
            icons=["file-text", "bar-chart", "x-circle", "list-ul", "linkedin"]
        )

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
    elif option == "LinkedIn Jobs":
        LinkedInScraper.main()

