import streamlit as st
import pandas as pd
import spacy
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Resume Ranking System",
    page_icon="📄",
    layout="wide"
)

# ------------------ LOAD NLP ------------------
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# ------------------ PREPROCESS ------------------
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return " ".join(tokens)

# ------------------ SKILLS ------------------
skills_list = [
    "python", "machine learning", "data analysis", "nlp",
    "tensorflow", "scikit-learn", "java", "sql", "deep learning"
]

def extract_skills(text):
    text = text.lower()
    return [skill for skill in skills_list if skill in text]

# ------------------ PDF TEXT EXTRACTION ------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)

    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()

    return text

# ------------------ UI HEADER ------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
    AI Resume Ranking System
    </h1>
    <p style='text-align: center;'>
    Upload resumes and rank candidates using AI
    </p>
    """,
    unsafe_allow_html=True
)

# ------------------ INPUTS ------------------
uploaded_files = st.file_uploader(
    "Upload PDF Resumes",
    type=["pdf"],
    accept_multiple_files=True
)

job_desc = st.text_area("Enter Job Description")

# ------------------ PROCESS ------------------
if st.button("Analyze Resumes"):

    if uploaded_files and job_desc.strip():

        job_clean = preprocess(job_desc)
        results = []

        progress = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):

            resume_text = extract_text_from_pdf(uploaded_file)

            if not resume_text.strip():
                continue

            resume_clean = preprocess(resume_text)

            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(
                [resume_clean, job_clean]
            )

            similarity = cosine_similarity(
                vectors[0:1],
                vectors[1:2]
            )

            score = similarity[0][0] * 100

            # Extract skills
            detected_skills = extract_skills(resume_text)

            results.append({
                "Resume": uploaded_file.name,
                "Score": round(score, 2),
                "Skills": ", ".join(detected_skills) if detected_skills else "None"
            })

            progress.progress((i + 1) / len(uploaded_files))

        if results:

            df = pd.DataFrame(results)
            df = df.sort_values(by="Score", ascending=False)

            # ------------------ OUTPUT ------------------
            st.subheader("📊 Resume Rankings")
            st.dataframe(df, use_container_width=True)

            st.subheader("📈 Candidate Scores")
            st.bar_chart(df.set_index("Resume"))

            best = df.iloc[0]

            st.success(
                f"🏆 Best Candidate: {best['Resume']} ({best['Score']}%)"
            )

        else:
            st.error("No readable content found in resumes.")

    else:
        st.warning("Please upload resumes and enter job description.")