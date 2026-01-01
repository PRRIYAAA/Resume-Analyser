import streamlit as st
from PyPDF2 import PdfReader
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import os

# ---------------- ENV ----------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

SYSTEM_PROMPT = """
You are an AI HR interviewer evaluating a fresher's resume.

Rules:
- Assume the candidate is a fresher
- Be encouraging and professional
- Do not reject the candidate
- Do not hallucinate experience

Provide detailed, actionable feedback.

Return output ONLY in this format:

Resume Score: <score>/100

Strengths:
- ...

Detailed Improvement Suggestions:
1. Resume Structure & Formatting:
- ...
2. Skills Section:
- ...
3. Projects / Academic Work:
- ...
4. ATS Optimization:
- ...
5. Communication & Language:
- ...

Role Readiness Level:
- Beginner / Intermediate / Job-Ready

Overall Advice:
- ...
"""

# ---------------- FUNCTIONS ----------------
def extract_text(file, filename):
    if filename.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return clean_text(text)

    if filename.endswith(".docx"):
        doc = docx.Document(file)
        text = "\n".join(p.text for p in doc.paragraphs)
        return clean_text(text)

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------- TF-IDF ATS ANALYSIS (ML) --------
def ats_analysis(resume_text, jd_text):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf = vectorizer.fit_transform([jd_text, resume_text])
    features = vectorizer.get_feature_names_out()

    jd_vec = tfidf[0].toarray()[0]
    resume_vec = tfidf[1].toarray()[0]

    similarity = cosine_similarity([jd_vec], [resume_vec])[0][0]
    ats_score = round(similarity * 100, 2)

    jd_keywords = sorted(
        zip(features, jd_vec),
        key=lambda x: x[1],
        reverse=True
    )[:25]

    resume_terms = set(resume_text.lower().split())
    matched, missing = [], []

    for word, weight in jd_keywords:
        if weight > 0:
            if word.lower() in resume_terms:
                matched.append(word)
            else:
                missing.append(word)

    return {
        "ats_score": ats_score,
        "matched_keywords": matched[:10],
        "missing_keywords": missing[:10]
    }

# -------- AI Resume Feedback --------
def analyze_resume_llm(resume_text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": resume_text}
        ]
    )
    return response.choices[0].message.content

# -------- ATS Rewrite Suggestions --------
def generate_rewrite_suggestions(missing_keywords, resume_text):
    if not missing_keywords:
        return "Your resume already aligns well with ATS keywords."

    keywords = ", ".join(missing_keywords[:8])

    prompt = f"""
You are an ATS optimization expert.

Missing keywords:
{keywords}

Suggest realistic resume rewrites WITHOUT inventing experience.

Resume Text:
{resume_text}

Return ONLY in this format:

Skills Section Rewrite:
- ...

Project Description Rewrite:
- ...

Experience / Internship Rewrite:
- ...

ATS Formatting Suggestions:
- ...
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
    )

    return response.choices[0].message.content

# ---------------- UI ----------------
st.set_page_config(
    page_title="AI Resume Analyzer & ATS Optimizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
    **AI Resume Analyzer** helps freshers optimize their resumes for better ATS compatibility and provides AI-powered feedback.
    
    **Features:**
    - 📊 AI Resume Scoring & Feedback
    - 🤖 ML-based ATS Matching
    - ✍️ Smart Rewrite Suggestions
    
    **Supported Formats:** PDF, DOCX
    """)
    st.markdown("---")
    st.caption("Built with Streamlit, OpenAI & Scikit-learn")

st.title("📄 AI Resume Analyzer & ATS Optimizer")
st.markdown("*Empower your career with AI-driven resume insights*")
st.markdown("---")

# Input Section
st.markdown("## 📥 Input Your Details")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Resume")
    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file",
        type=["pdf", "docx"],
        help="Upload your resume in PDF or DOCX format for analysis."
    )

with col2:
    st.markdown("### 📌 Job Description")
    job_description = st.text_area(
        "Paste the job description here (optional for ATS matching)",
        height=150,
        help="Provide a job description to get ATS matching score and keyword suggestions."
    )

# Analyze Button
st.markdown("---")
analyze_btn = st.button(
    "🚀 Analyze Resume",
    use_container_width=True,
    type="primary",
    help="Click to start the AI analysis of your resume."
)

if analyze_btn:
    if uploaded_file is None:
        st.error("⚠️ Please upload a resume to proceed.")
    else:
        with st.spinner("🔍 Analyzing your resume with AI and ML... Please wait."):
            resume_text = extract_text(uploaded_file, uploaded_file.name)
            ai_feedback = analyze_resume_llm(resume_text)

        st.success("✅ Analysis completed successfully!")

        # Results in Tabs
        tab1, tab2, tab3 = st.tabs(["📊 AI Feedback", "🤖 ATS Analysis", "✍️ Rewrite Suggestions"])

        with tab1:
            st.markdown("### AI-Powered Resume Evaluation")
            st.markdown(ai_feedback)

        with tab2:
            if job_description.strip():
                ats = ats_analysis(resume_text, job_description)
                st.markdown("### ML-Based ATS Matching")
                
                # Score Display
                score_col, status_col = st.columns([1, 2])
                with score_col:
                    st.metric("ATS Match Score", f"{ats['ats_score']}%")
                with status_col:
                    if ats["ats_score"] < 40:
                        st.error("🔴 Low ATS Match - Needs Improvement")
                    elif ats["ats_score"] < 65:
                        st.warning("🟡 Moderate ATS Match - Room for Enhancement")
                    else:
                        st.success("🟢 Strong ATS Match - Great Job!")
                
                # Keywords
                kw_col1, kw_col2 = st.columns(2)
                with kw_col1:
                    st.markdown("#### ✅ Matched Keywords")
                    if ats["matched_keywords"]:
                        for kw in ats["matched_keywords"]:
                            st.markdown(f"- {kw}")
                    else:
                        st.write("No matches found.")
                
                with kw_col2:
                    st.markdown("#### ❌ Missing Keywords")
                    if ats["missing_keywords"]:
                        for kw in ats["missing_keywords"]:
                            st.markdown(f"- {kw}")
                    else:
                        st.write("All keywords matched!")
            else:
                st.info("💡 Provide a job description to see ATS analysis.")

        with tab3:
            if job_description.strip():
                ats = ats_analysis(resume_text, job_description)
                rewrite = generate_rewrite_suggestions(
                    ats["missing_keywords"],
                    resume_text
                )
                st.markdown("### ATS-Optimized Rewrite Suggestions")
                st.markdown(rewrite)
            else:
                st.info("💡 Provide a job description to get rewrite suggestions.")

st.markdown("---")
st.caption("🔧 Powered by AI & ML | Designed for Freshers | TF-IDF ATS Matching")
