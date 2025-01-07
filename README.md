# Resume Parser Using Gemini

## Overview

The Resume Parser is a powerful tool designed to automate the extraction and analysis of key information from resumes. By leveraging the Gemini API, this application converts unstructured resume data into structured formats. Using RAG technique, personalized AI-based recommendation is done on several aspects including course suggestions.

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
Enter your GEMINI API KEY, LANGCHAIN API KEY in the .env file in the root directory. Also, Location of the JSON file of GOOGLE APPLICATION CREDENTIALS need to be given.
```
### 5. Run the Flask App
```
flask --app main run
```


