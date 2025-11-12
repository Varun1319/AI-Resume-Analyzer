import time
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from openai import OpenAI
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Resume Analyzer AI", layout="wide")


def pdf_to_chunks(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for p in reader.pages:
        t = p.extract_text()
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


def make_client(key):
    if not key:
        raise ValueError("OpenAI API key missing")
    return OpenAI(api_key=key)


def get_embeddings(client, texts, model="text-embedding-3-small"):
    resp = client.embeddings.create(model=model, input=texts)
    vectors = [np.array(item.embedding, dtype=float) for item in resp.data]
    return np.vstack(vectors)


def cosine_sim_matrix(mat, vec):
    if mat.size == 0 or vec.size == 0:
        return np.array([])
    mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    vec_norm = vec / (np.linalg.norm(vec) + 1e-12)
    return np.dot(mat_norm, vec_norm)


def openai_analyze(openai_api_key, chunks, analyze, top_k=3, chat_model="gpt-3.5-turbo", embed_model="text-embedding-3-small"):
    client = make_client(openai_api_key)
    chunk_vecs = get_embeddings(client, chunks, model=embed_model)
    query_vec = get_embeddings(client, [analyze], model=embed_model)[0]
    sims = cosine_sim_matrix(chunk_vecs, query_vec)
    if sims.size == 0:
        top_idx = np.arange(min(top_k, len(chunks)))
    else:
        top_idx = np.argsort(-sims)[:top_k]

    selected = [chunks[i] for i in top_idx if i < len(chunks)]
    context = "\n\n---\n\n".join(selected)

    # ✅ Proper indentation: this stays INSIDE the function
    prompt = f"""
You are an assistant that carefully analyzes resumes. Use the provided resume excerpts and then answer the user's instruction below.

Context excerpts:
\"\"\"{context}\"\"\"

User instruction:
{analyze}

Please answer clearly and concisely, and include any recommendations when relevant.
"""

    messages = [{"role": "user", "content": prompt}]
    resp = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=0.0,
        max_tokens=900
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)


def summary_prompt(chunks):
    return f"Provide a detailed summary and a short conclusion for these resume excerpts:\n{chunks}"


def strength_prompt(chunks):
    return f"Analyze and explain the strengths in these resume excerpts:\n{chunks}"


def weakness_prompt(chunks):
    return f"Analyze weaknesses and provide actionable improvements for these resume excerpts:\n{chunks}"


def job_titles_prompt(chunks):
    return f"Suggest suitable job titles for this candidate based on these resume excerpts:\n{chunks}"


def main():
    st.title("Resume Analyzer AI (OpenAI)")
    with st.sidebar:
        add_vertical_space(2)
        option = option_menu(
            menu_title="",
            options=["Summary", "Strength", "Weakness", "Job Titles"],
            icons=["file-text", "star", "exclamation-circle", "list"],
            default_index=0
        )

    if option == "Summary":
        with st.form("SummaryForm"):
            pdf = st.file_uploader("Upload Your Resume (PDF)", type="pdf")
            openai_api_key = st.text_input("Enter OpenAI API Key", type="password", key="summary_key")
            submit = st.form_submit_button("Submit")
        if submit:
            if pdf and openai_api_key:
                with st.spinner("Processing..."):
                    chunks = pdf_to_chunks(pdf)
                    prompt_text = summary_prompt(' '.join(chunks[:6]))
                    try:
                        out = openai_analyze(openai_api_key, chunks, prompt_text)
                        st.header("Summary")
                        st.write(out)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please upload a resume and enter your OpenAI API key.")

    elif option == "Strength":
        with st.form("StrengthForm"):
            pdf = st.file_uploader("Upload Your Resume (PDF)", type="pdf")
            openai_api_key = st.text_input("Enter OpenAI API Key", type="password", key="strength_key")
            submit = st.form_submit_button("Submit")
        if submit:
            if pdf and openai_api_key:
                with st.spinner("Processing..."):
                    chunks = pdf_to_chunks(pdf)
                    prompt_text = strength_prompt(' '.join(chunks[:6]))
                    try:
                        out = openai_analyze(openai_api_key, chunks, prompt_text)
                        st.header("Strengths")
                        st.write(out)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please upload a resume and enter your OpenAI API key.")

    elif option == "Weakness":
        with st.form("WeakForm"):
            pdf = st.file_uploader("Upload Your Resume (PDF)", type="pdf")
            openai_api_key = st.text_input("Enter OpenAI API Key", type="password", key="weakness_key")
            submit = st.form_submit_button("Submit")
        if submit:
            if pdf and openai_api_key:
                with st.spinner("Processing..."):
                    chunks = pdf_to_chunks(pdf)
                    prompt_text = weakness_prompt(' '.join(chunks[:6]))
                    try:
                        out = openai_analyze(openai_api_key, chunks, prompt_text)
                        st.header("Weaknesses & Suggestions")
                        st.write(out)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please upload a resume and enter your OpenAI API key.")

    elif option == "Job Titles":
        with st.form("JobForm"):
            pdf = st.file_uploader("Upload Your Resume (PDF)", type="pdf")
            openai_api_key = st.text_input("Enter OpenAI API Key", type="password", key="job_key")
            submit = st.form_submit_button("Submit")
        if submit:
            if pdf and openai_api_key:
                with st.spinner("Processing..."):
                    chunks = pdf_to_chunks(pdf)
                    prompt_text = job_titles_prompt(' '.join(chunks[:6]))
                    try:
                        out = openai_analyze(openai_api_key, chunks, prompt_text)
                        st.header("Suggested Job Titles")
                        st.write(out)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please upload a resume and enter your OpenAI API key.")


if __name__ == "__main__":
    main()
