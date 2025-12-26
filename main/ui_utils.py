# ui_utils.py
import streamlit as st

def streamlit_config():
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
