# resume_analyzer.py
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
import warnings

warnings.filterwarnings("ignore")
class ResumeAnalyzer:
    @staticmethod
    def pdf_to_chunks(pdf) -> list[str]:
        """
        Convert a PDF resume into overlapping text chunks for analysis.
        """
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
    def _make_client() -> OpenAI:
        key = getattr(st.session_state, "openai_api_key", None)

        if not key:
            st.warning("⚠️ Please paste your OpenAI API key in the box above.")
            st.stop()

        # Debug: show prefix so we see what is actually used
        st.write("DEBUG – Using API key starting with:", str(key)[:10])

        return OpenAI(api_key=key)

    @staticmethod
    def _get_embeddings(client: OpenAI, texts, model: str = "text-embedding-3-small"):
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

    @staticmethod
    def openai_analyze(chunks, prompt, top_k: int = 3) -> str:
        """
        Retrieve top-k relevant chunks using embeddings, then ask OpenAI to analyze.
        """
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

    # ---------- Prompt builders ---------- #
    @staticmethod
    def summary_prompt(chunks_str: str) -> str:
        return (
            "Provide a **comprehensive and well-structured summary** of this resume. "
            "Include key skills, educational highlights, technical expertise, and achievements. "
            "Also add a concluding note on overall candidate suitability. "
            f"Resume content:\n\n{chunks_str}"
        )

    @staticmethod
    def strength_prompt(chunks_str: str) -> str:
        return (
            "Analyze this resume and write a **detailed list of the candidate strengths**. "
            "Include technical strengths, soft skills, leadership abilities, teamwork, achievements, "
            "and any unique differentiating factors. Give **short examples** from the text wherever possible. "
            f"Resume content:\n\n{chunks_str}"
        )

    @staticmethod
    def weakness_prompt(chunks_str: str) -> str:
        return (
            "Critically analyze this resume to identify **weak areas or gaps** in terms of formatting, "
            "missing keywords, or lack of quantifiable results. Suggest **specific improvements** to make "
            "it more impactful for recruiters or ATS. "
            f"Resume content:\n\n{chunks_str}"
        )

    @staticmethod
    def job_title_prompt(chunks_str: str) -> str:
        return (
            "Based on this resume, list **5–10 most suitable job roles or titles**, categorized by domain "
            "(e.g., Software Engineering, Data, Product, etc.) and include **a one-line reason** why each "
            "fits the profile. "
            f"Resume content:\n\n{chunks_str}"
        )

    @staticmethod
    def run_all(chunks: list[str]):
        """
        Run all analyses (summary, strengths, weaknesses, job titles) and
        store results into Streamlit session_state.
        """
        st.info("⏳ Generating all insights (summary, strengths, weaknesses, titles)...")

        # Use only first few chunks to keep prompts compact
        base_text = " ".join(chunks[:6])

        summary = ResumeAnalyzer.openai_analyze(
            chunks, ResumeAnalyzer.summary_prompt(base_text)
        )
        strength = ResumeAnalyzer.openai_analyze(
            chunks, ResumeAnalyzer.strength_prompt(base_text)
        )
        weakness = ResumeAnalyzer.openai_analyze(
            chunks, ResumeAnalyzer.weakness_prompt(base_text)
        )
        job_titles = ResumeAnalyzer.openai_analyze(
            chunks, ResumeAnalyzer.job_title_prompt(base_text)
        )

        st.session_state["resume_summary"] = summary
        st.session_state["resume_strength"] = strength
        st.session_state["resume_weakness"] = weakness
        st.session_state["resume_jobtitles"] = job_titles

        st.success("✅ All resume insights generated! You can navigate using the sidebar.")
