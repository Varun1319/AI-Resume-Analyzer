# ui/common.py
from io import BytesIO

import streamlit as st

from resume_analyzer.documents import pdf_to_chunks


def apply_page_style():
    # Basic CSS styling and title
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


@st.cache_data(show_spinner=False)
def chunks_from_pdf_bytes(file_bytes: bytes) -> list[str]:
    return pdf_to_chunks(BytesIO(file_bytes))
