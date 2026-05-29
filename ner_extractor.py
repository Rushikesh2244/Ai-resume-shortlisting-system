"""
utils/ner_extractor.py
Named Entity Recognition + pattern-based extraction of structured fields
from raw resume text: name, email, phone, skills, education, experience years.
"""

import re
from typing import Dict, List, Any

# ── Comprehensive tech skills vocabulary ──────────────────────────────────────
SKILLS_VOCAB = {
    # Languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "scala",
    "r", "go", "rust", "swift", "kotlin", "ruby", "php", "matlab", "bash",
    # ML / AI
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "reinforcement learning", "generative ai",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "xgboost",
    "lightgbm", "catboost", "transformers", "bert", "gpt", "llm",
    "hugging face", "spacy", "nltk", "gensim",
    # Data
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "tableau", "power bi",
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "apache spark", "hadoop", "kafka", "airflow", "dbt", "etl",
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform",
    "ci/cd", "jenkins", "github actions", "linux",
    # Web / API
    "flask", "fastapi", "django", "react", "node.js", "rest api", "graphql",
    # Concepts
    "tf-idf", "cosine similarity", "word2vec", "embeddings",
    "named entity recognition", "ner", "text classification",
    "recommendation system", "collaborative filtering",
    # Misc
    "git", "agile", "scrum", "figma", "excel",
}

EDUCATION_KEYWORDS = {
    "b.tech", "b.e", "b.sc", "b.com", "bca", "bba",
    "m.tech", "m.sc", "mca", "mba", "m.e",
    "phd", "ph.d", "doctorate",
    "bachelor", "master", "degree", "diploma",
}

DEGREE_LEVELS = {
    "phd": 5, "ph.d": 5, "doctorate": 5,
    "m.tech": 4, "m.sc": 4, "mca": 4, "mba": 4, "master": 4, "m.e": 4,
    "b.tech": 3, "b.e": 3, "b.sc": 3, "bachelor": 3, "bca": 3, "bba": 3,
    "diploma": 2,
}


# ── Regex patterns ─────────────────────────────────────────────────────────────

EMAIL_RE    = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PHONE_RE    = re.compile(r"(\+?\d[\d\s\-().]{7,14}\d)")
NAME_RE     = re.compile(r"(?:name\s*[:\-]?\s*)([A-Z][a-z]+(?: [A-Z][a-z]+)+)", re.IGNORECASE)
GPA_RE      = re.compile(r"(?:gpa|cgpa|score)\s*[:\-]?\s*(\d+\.?\d*)\s*/\s*(\d+\.?\d*)", re.IGNORECASE)
YEAR_RE     = re.compile(r"\b(19|20)\d{2}\b")


def extract_email(text: str) -> str:
    match = EMAIL_RE.search(text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    match = PHONE_RE.search(text)
    return match.group(0).strip() if match else ""


def extract_name(text: str) -> str:
    """Try labeled pattern first, then first capitalized full name in text."""
    match = NAME_RE.search(text)
    if match:
        return match.group(1).strip()
    # Fallback: first line that looks like a proper name
    for line in text.splitlines():
        line = line.strip()
        parts = line.split()
        if 2 <= len(parts) <= 4 and all(p[0].isupper() for p in parts if p):
            if not any(kw in line.lower() for kw in ["resume", "curriculum", "vitae", "profile"]):
                return line
    return "Unknown"


def extract_skills(text: str) -> List[str]:
    """Match skills vocabulary against the resume text (case-insensitive)."""
    text_lower = text.lower()
    found = []
    for skill in sorted(SKILLS_VOCAB):
        # Use word-boundary-aware search
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


def extract_education(text: str) -> Dict[str, Any]:
    """Extract highest degree level and GPA if present."""
    text_lower = text.lower()
    highest_level = 0
    degree_name = ""

    for keyword, level in DEGREE_LEVELS.items():
        if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
            if level > highest_level:
                highest_level = level
                degree_name = keyword.upper()

    gpa_match = GPA_RE.search(text)
    gpa = None
    if gpa_match:
        score = float(gpa_match.group(1))
        total = float(gpa_match.group(2))
        gpa = round((score / total) * 10, 2) if total != 10 else score  # normalise to /10

    return {
        "degree": degree_name,
        "degree_level": highest_level,
        "gpa": gpa,
    }


def extract_experience_years(text: str) -> float:
    """
    Estimate total years of experience from date ranges like '2020 - 2023'
    or 'Present'. Returns 0 if nothing found.
    """
    present_year = 2026
    total_years = 0.0

    # Pattern: YYYY - YYYY  or  YYYY - Present
    ranges = re.findall(
        r"(20\d{2}|19\d{2})\s*[-–—to]+\s*(20\d{2}|19\d{2}|present|current|now)",
        text.lower()
    )
    for start_str, end_str in ranges:
        start = int(start_str)
        end = present_year if end_str in ("present", "current", "now") else int(end_str)
        if end >= start:
            total_years += (end - start)

    return round(total_years, 1)


def extract_all(text: str, filename: str = "") -> Dict[str, Any]:
    """
    Run all extractors on raw resume text.
    Returns a structured dictionary of parsed fields.
    """
    name = extract_name(text)
    if name == "Unknown" and filename:
        # Use filename as fallback name
        base = re.sub(r"[_\-]", " ", filename.replace(".txt", "").replace(".pdf", "").replace(".docx", ""))
        name = base.title()

    return {
        "name":             name,
        "email":            extract_email(text),
        "phone":            extract_phone(text),
        "skills":           extract_skills(text),
        "education":        extract_education(text),
        "experience_years": extract_experience_years(text),
    }
