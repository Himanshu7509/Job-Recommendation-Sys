import re
import phonenumbers
from app.utils.file_reader import read_resume_from_upload
from app.services.skill_matcher import nlp, matcher

def extract_skills_from_doc(doc):
    # We now receive a doc instead of text
    matches = matcher(doc)
    skills = set()
    for _, start, end in matches:
        skill = doc[start:end].text.lower().strip()
        skill = re.sub(r"[^a-zA-Z0-9+.# ]", "", skill)
        if len(skill) > 1:
            skills.add(skill)
    return sorted(skills)


def extract_experience_years(text):
    patterns = [
        r"(\d+)\+?\s*years",
        r"(\d+)\+?\s*yrs",
        r"experience\s*[:\-]?\s*(\d+)",
        r"over\s*(\d+)\s*years"
    ]

    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        years.extend([int(m) for m in matches])

    return max(years) if years else None

def extract_education_from_doc(doc):
    degrees = ["bachelor","b.tech","be","b.sc","master","m.tech","mba","phd","diploma"]
    results = []
    for sent in doc.sents:
        s = sent.text.lower()
        if any(deg in s for deg in degrees):
            results.append(sent.text.strip())
    return results

def extract_projects_from_doc(doc):
    projects = []
    text_lower = doc.text.lower()
    if "project" in text_lower:
        # Fallback to text for slicing but could be optimized further
        section = text_lower.split("project", 1)[1][:2000]
        temp_doc = nlp(section)
        for sent in temp_doc.sents:
            if len(sent.text.strip()) > 20:
                projects.append(sent.text.strip())
    return projects[:5]


def extract_email(text):
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else None


def extract_phone(text):
    for match in phonenumbers.PhoneNumberMatcher(text, None):
        return phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
    return None

def extract_name_from_doc(doc):
    # Use the first 300 tokens of the already processed doc
    for ent in doc[:100].ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def parse_resume(text):
    doc = nlp(text)
    return {
        "name": extract_name_from_doc(doc),
        "skills": extract_skills_from_doc(doc),
        "experience_years": extract_experience_years(text),
        "education": extract_education_from_doc(doc),
        "projects": extract_projects_from_doc(doc),
        "email": extract_email(text),
        "phone": extract_phone(text)
    }


def clean_text(text):
    text = re.sub(r"[^\x20-\x7E]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def parse_resume_file(file):
    raw = read_resume_from_upload(file)
    return clean_text(raw)