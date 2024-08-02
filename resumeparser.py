import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get('GEMINI_API_KEY')

# Check if API key is loaded
if not api_key:
    raise ValueError("API key is missing. Please set 'GEMINI_API_KEY' in your .env file.")

# Configure Gemini API
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    raise RuntimeError(f"Failed to configure Gemini API: {e}")

def ats_extractor(resume_data):
    """
    Extracts information from a resume using Gemini

    Args:
        resume_data: String containing the resume content

    Returns:
        dict: Extracted information in JSON format
    """
    
    prompt = """
    You are an AI trained to extract information from resumes. Given a resume, identify and return the following details in JSON format:

    1. Full Name
    2. Email Address and Phone Number (an extra 'pe' is coming at the start in some cases. Ignore irrelevant 'pe' at start of email address.)
    3. GitHub Profile URL
    4. LinkedIn Profile URL
    5. Education Details (Degree, Institution, Dates, Percentage / CGPA [Only include those details which are given in the resume])
    6. Employment History (Company, Position, Dates)
    7. Project Details in Short (Ignore the word 'Demo', it is used to indicate Demo Link)
    8. Certifications
    9. Achievements and Awards
    10. Technical Skills
    11. Soft Skills
    12. Other Relevant Details
    
    Extract the skills and relevant keywords to recommend the best job roles for the candidate. Give only the top 3 most matching job roles.
    
    Recommend any certifications or courses that the candidate can take to improve their skills. Give maximum of 5 recommendations.
    
    Note:- Just give a JSON file, no need to give any explanation. Remove extra 'pe' from email address. Do not need to number anything.
    """

    try:
        response = model.generate_content(prompt + resume_data)
        response_text = response.text.strip()
        
        # Handle the response format
        if response_text.startswith("```json"):
            response_text = response_text[7:].strip()  
        if response_text.endswith("```"):
            response_text = response_text[:-3].strip()
        
        # Print and return the cleaned response
        print(response_text)
        return remove_empty_brackets(response_text)
    except Exception as e:
        print(f"Error during API request: {e}")
        return None

def remove_empty_brackets(input_string):
    lines = input_string.splitlines()
    filtered_lines = [line for line in lines if '[]' not in line]
    return '\n'.join(filtered_lines)
