import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import re
from fuzzywuzzy import fuzz  # LLM Concept: Fuzzy matching (approximate string matching)
from nltk.stem import WordNetLemmatizer  # LLM Concept: Text preprocessing with NLP techniques

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    text = ""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")  # Read uploaded file
        for page in doc:
            text += page.get_text("text") + " "
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text.strip()

# Function to clean and normalize text (Preprocessing for LLM-style text analysis)
def clean_text(text):
    """Cleans and normalizes text for better matching."""
    text = text.lower().strip()  # Convert to lowercase  # LLM Concept: Case normalization
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces  # LLM Concept: Text preprocessing
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split()]  # LLM Concept: Lemmatization for word standardization
    return ' '.join(words)

# Function to check keyword match and highlight detected skills
def check_keywords(cv_text, job_keywords):
    """Matches CV skills with job description keywords using fuzzy matching."""
    matched_skills = []
    missing_skills = []
    
    for keyword in job_keywords:
        keyword_clean = clean_text(keyword)

        # Direct check if the skill exists in CV
        if keyword_clean in cv_text:
            matched_skills.append(keyword)
        else:
            # Try fuzzy matching for approximate matches
            match_score = fuzz.partial_ratio(cv_text, keyword_clean)  # LLM Concept: Approximate text matching using NLP
            if match_score > 80:  # 80% similarity threshold
                matched_skills.append(keyword)
            else:
                missing_skills.append(keyword)

    return matched_skills, missing_skills

# Function to highlight matched skills in CV text
def highlight_skills(cv_text, matched_skills):
    """Highlights detected skills in yellow."""
    for skill in matched_skills:
        cv_text = re.sub(f"(?i)({re.escape(skill)})", r'<mark style="background-color: yellow">\1</mark>', cv_text)  # LLM Concept: Regex-based text modification
    return cv_text

# Streamlit UI
st.title("📄 AI CV Analyzer")

# File uploader for CV
uploaded_file = st.file_uploader("Upload your CV (PDF only)", type=["pdf"])

# User input for required skills
job_description = st.text_area("Enter Required Skills (comma-separated)")

# Button to start analysis
if st.button("Analyze CV"):
    if uploaded_file is None:
        st.error("❌ Please upload a CV.")
    elif not job_description.strip():
        st.error("❌ Please enter required skills.")
    else:
        # Extract text from CV
        cv_text = extract_text_from_pdf(uploaded_file)
        cv_text_clean = clean_text(cv_text)  # LLM Concept: Text normalization before NLP processing

        # Extract keywords from job description
        job_keywords = [kw.strip() for kw in job_description.split(",") if kw.strip()]

        # Check for matching and missing skills
        matched_skills, missing_skills = check_keywords(cv_text_clean, job_keywords)

        # Highlight matched skills in CV text
        highlighted_cv_text = highlight_skills(cv_text, matched_skills)  # LLM Concept: Dynamic text modification

        # Display results
        st.subheader("🔍 CV Analysis Results")

        if matched_skills:
            st.success(f"✅ Skills in your CV: {', '.join(matched_skills)}")
        else:
            st.warning("⚠️ No matching skills found.")

        if missing_skills:
            st.warning(f"⚠️ Missing Skills: {', '.join(missing_skills)}")
        else:
            st.success("🎉 No missing skills!")

        # Show highlighted extracted CV text for debugging
        with st.expander("📄 View Extracted CV Text (Debugging)"):
            st.markdown(highlighted_cv_text, unsafe_allow_html=True)  # LLM Concept: Rendering modified text with HTML
