import os
import sys
import json
from flask import Flask, request, render_template
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
from google.cloud import aiplatform
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Configure the UPLOAD_PATH for saving uploaded files
UPLOAD_PATH = r"__DATA__"
app = Flask(__name__)

# Configure the path for custom modules
sys.path.insert(0, os.path.abspath(os.getcwd()))

# Configuring credentials
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    raise ValueError("API key is missing. Please set 'GEMINI_API_KEY' in your .env file.")
google_credentials = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
if google_credentials:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials

# Initialize the Vertex AI platform
try:
    aiplatform.init(project='central-segment-447015-f3', location='us-central1')  # Your Google Cloud project details
    print("Google Cloud AI Platform initialized successfully!")
except Exception as e:
    raise RuntimeError(f"Failed to initialize Google Cloud AI platform: {e}")

# Configure Gemini API
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    raise RuntimeError(f"Failed to configure Gemini API: {e}")

# Load the knowledge base CSV containing courses and certifications
knowledge_base_df = pd.read_csv('knowledge_base.csv', on_bad_lines='skip')


# Function to extract information from the resume using Gemini
def ats_extractor_with_rag(resume_data):
    # Extract relevant keywords from resume (e.g., skills, job roles)
    extracted_keywords = extract_keywords(resume_data)

    # Retrieve the relevant courses and certifications based on extracted keywords
    relevant_courses = get_relevant_courses(extracted_keywords)

    # Create the combined prompt with resume data and course information
    combined_prompt = f"""
    Given the following resume data and relevant course/certification recommendations:

    Resume Data:
    {resume_data}

    Relevant Recommendations (Courses, Certifications, etc.):
    {relevant_courses}

    Based on the above, generate recommendations for career improvement and certifications.
    """
    
    # Generate the response from the Gemini API
    response = model.generate_content(combined_prompt)
    return response.text.strip()

# Function to extract keywords (skills, job roles, etc.) from resume
def extract_keywords(resume_data):
    keywords = resume_data.split()
    return keywords

# Function to retrieve relevant courses based on extracted keywords
def get_relevant_courses(extracted_keywords):
    vectorizer = TfidfVectorizer(stop_words='english')

    # Combine all the relevant course information from the knowledge base into a single text
    course_descriptions = knowledge_base_df['Title'] + " " + knowledge_base_df['Skills']
    
    # Fit the vectorizer on the course descriptions
    vectorizer.fit(course_descriptions)
    
    # Transform the extracted keywords into the same vector space
    resume_keywords = [" ".join(extracted_keywords)]
    resume_vec = vectorizer.transform(resume_keywords)
    
    # Calculate cosine similarity between resume and course descriptions
    course_vecs = vectorizer.transform(course_descriptions)
    cosine_similarities = np.dot(resume_vec, course_vecs.T).toarray().flatten()

    # Sort courses by similarity and get the top 5
    top_course_indices = cosine_similarities.argsort()[-5:][::-1]
    
    # Retrieve the top matching courses
    relevant_courses = knowledge_base_df.iloc[top_course_indices][['Title', 'Provider', 'Type', 'Duration']]
    
    return relevant_courses.to_dict(orient='records')


# Function to read the PDF and extract text
def _read_file_from_path(path):
    reader = PdfReader(path) 
    data = ""

    for page_no in range(len(reader.pages)):
        page = reader.pages[page_no] 
        data += page.extract_text()

    return data 

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/process", methods=["POST"])
def ats():
    if 'pdf_doc' not in request.files:
        return render_template('index.html', error="No PDF file uploaded. Please upload a valid PDF.")

    doc = request.files['pdf_doc']
    
    if doc.filename == '':
        return render_template('index.html', error="No file selected. Please select a PDF file to upload.")

    # Save the uploaded PDF file
    doc.save(os.path.join(UPLOAD_PATH, "file.pdf"))
    doc_path = os.path.join(UPLOAD_PATH, "file.pdf")
    
    # Read the file and extract information
    resume_data = _read_file_from_path(doc_path)
    
    # Extract resume details and get course recommendations
    response = ats_extractor_with_rag(resume_data)
    
    print("Processed Resume Data:")
    print(resume_data) 

    print("Generated Course and Certification Recommendations:")
    print(response)
    
    return render_template('index.html', data=json.dumps({"Resume Data": resume_data, "AI-based Recommendations": response}))

if __name__ == "__main__":
    app.run(port=8000, debug=True)
