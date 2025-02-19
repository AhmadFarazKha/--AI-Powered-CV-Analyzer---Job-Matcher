import re
import nltk
import torch
import docx
from PyPDF2 import PdfReader
from transformers import pipeline  # LLM Concept: Using pre-trained Transformer models for NLP tasks

nltk.download("punkt")  # LLM Concept: Downloading NLTK tokenizer for text processing

# Load Hugging Face NER model
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")  # LLM Concept: Named Entity Recognition (NER) using BERT

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_file)  # LLM Concept: Processing structured text from a PDF
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text.strip()

def extract_text_from_docx(docx_file):
    """Extracts text from a DOCX file."""
    doc = docx.Document(docx_file)  # LLM Concept: Processing text from a Word document
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def extract_skills_from_text(text):
    """Extracts skills using Named Entity Recognition (NER)."""
    ner_results = ner_pipeline(text)  # LLM Concept: Using an NER model to extract relevant named entities
    skills = [entity["word"] for entity in ner_results if entity["entity"] == "B-MISC"]  # LLM Concept: Identifying skills tagged as 'B-MISC'
    return list(set(skills))

def extract_keywords_from_job_desc(job_desc):
    """Extracts important keywords from the job description."""
    words = nltk.word_tokenize(job_desc)  # LLM Concept: Tokenizing text using NLP techniques
    keywords = [word for word in words if word.isalpha()]  # LLM Concept: Filtering out non-alphabetic tokens
    return list(set(keywords))

def match_skills(cv_skills, job_keywords):
    """Matches CV skills with job description keywords."""
    missing_skills = [skill for skill in job_keywords if skill not in cv_skills]  # LLM Concept: Matching skills using direct keyword comparison
    return missing_skills

def rephrase_text(text):
    """Rephrases text using Hugging Face text generation."""
    summarizer = pipeline("text2text-generation", model="facebook/bart-large-cnn")  # LLM Concept: Using BART for text summarization/rephrasing
    return summarizer(text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']  # LLM Concept: Generating rephrased text using a transformer model
