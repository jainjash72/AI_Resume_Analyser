import streamlit as st
from backend.pdf_ingestion import load_split_pdf
from backend.vector_store import create_vector_store
from backend.analysis import analyze_resume
import os
import shutil

def render_main_app():
    # Custom CSS to adjust the sidebar width
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            min-width: 25%;
            max-width: 25%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.header("Upload Resume")
        resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
        job_description = st.text_area("Enter Job Description", height=300)

        if resume_file and job_description:
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            with open(os.path.join(temp_dir, resume_file.name), "wb") as f:
                f.write(resume_file.getbuffer())
            
            resume_file_path = os.path.join("temp", resume_file.name)
            resume_docs, resume_chunks = load_split_pdf(resume_file_path)
            vector_store = create_vector_store(resume_chunks)
            st.session_state.vector_store = vector_store
            shutil.rmtree(temp_dir)

            if st.button("Analyze Resume", help="Click to analyze the resume"):
                full_resume = " ".join([doc.page_content for doc in resume_docs])
                analysis = analyze_resume(full_resume, job_description)
                st.session_state.analysis = analysis    
        else:
            st.info("Please upload a resume and enter a job description to begin.")

    if "analysis" in st.session_state:
        st.header("Resume-Job Compatibility Analysis")
        st.write(st.session_state.analysis)
    else:
        st.header("Welcome to the Ultimate Resume Analysis Tool!")
        st.subheader("Your one-stop solution for resume screening and analysis.")
        st.info("Do you want to find out the compatibility between a resume and a job description? So what are you waiting for?")
        todo = ["Upload a Resume", "Enter a Job Description", "Click on Analyze Resume"]
        st.markdown("\n".join([f"##### {i+1}. {item}" for i, item in enumerate(todo)]))
