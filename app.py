import streamlit as st
from frontend.main_app import render_main_app
from frontend.chat_interface import render_chat_interface

st.set_page_config(layout="wide")

def main():
    st.title("Resume Analyzer")
    with st.sidebar:
        st.image("resume_analyzer_logo.png", width=150)

    col1, col2 = st.columns([3, 2])
    with col1:
        render_main_app()
    with col2:
        render_chat_interface()

if __name__ == "__main__":
    main()
