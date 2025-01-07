# Resume Parser Using Gemini

## Overview

The Resume Parser is a powerful tool designed to automate the extraction and analysis of key information from resumes. By leveraging the Gemini API, this application converts unstructured resume data into structured formats. Using the RAG technique with Google’s Vertex AI, personalized AI-based recommendations are made across various aspects, including course suggestions.

## Steps to Run the App (Locally) :-

### 1. Clone the repository:
```
git clone https://github.com/GDGouravDey/Resume-Parser-Gemini.git
cd Resume-Parser-Gemini
```
### 2. Set up a virtual environment:
```
python -m venv venv
source venv/bin/activate
On Windows, use `venv\Scripts\activate`
```
### 3. Install the dependencies:
```
pip install -r requirements.txt
```
### 4. Configure environment variables:
```
I. GEMINI API Key: Enter your GEMINI API key in the .env file located in the root directory.
II. LANGCHAIN API Key: Provide your LANGCHAIN API key in the .env file.
III. Google Application Credentials: Specify the location of your Google Application Credentials JSON file in the .env file.
```
### 5. Run the Flask App
```
flask --app main run
```


