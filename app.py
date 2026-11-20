import streamlit as st
from streamlit_option_menu import option_menu

from resume_analyzer.ui import candidate, linkedin, recruiter
from resume_analyzer.ui.common import apply_page_style

# ✅ must be first Streamlit call
st.set_page_config(page_title="Resume Analyzer AI", layout="wide")


# ---------------------- MAIN APP ---------------------- #
def main():
    apply_page_style()
    st.write("")
    st.write("")

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
                options=[*candidate.PAGES, "LinkedIn Jobs"],
                icons=["file-text", "bar-chart", "x-circle", "list-ul", "speedometer", "linkedin"],
            )
        else:
            option = option_menu(
                menu_title="Recruiter Mode",
                options=["Multi-Resume Ranking"],
                icons=["layers"],
            )

    # ---- MODE INFO BANNER ----
    if "Candidate" in mode:
        st.info("👤 Candidate Mode – Analyze and improve a single resume.")
    else:
        st.info("🧑‍💼 Recruiter Mode – Rank and inspect multiple candidates.")

    # ---- PAGE LOGIC ----
    if option == "Multi-Resume Ranking":
        recruiter.render()
    elif option == "LinkedIn Jobs":
        linkedin.render()
    elif option in candidate.PAGES:
        candidate.render(option)


if __name__ == "__main__":
    main()
