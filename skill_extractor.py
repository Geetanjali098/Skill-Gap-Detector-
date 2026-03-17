# skill_extractor.py
# This module extracts skills from text using NLP and a predefined skill list.

import spacy
import re

# -------------------------------------------------------
# Predefined list of common tech & soft skills to detect
# -------------------------------------------------------
SKILLS_DB = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "swift",
    "kotlin", "go", "rust", "scala", "r", "matlab", "php", "perl", "bash",
    # Web Development
    "html", "css", "react", "angular", "vue", "node.js", "django", "flask",
    "fastapi", "spring boot", "express", "next.js", "nuxt", "tailwind",
    # Data & ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "data analysis",
    "data science", "feature engineering", "model deployment",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "sqlite", "oracle", "nosql", "dynamodb", "firebase",
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "ci/cd", "jenkins", "github actions", "terraform", "ansible",
    "linux", "unix", "shell scripting",
    # Tools & Practices
    "git", "github", "gitlab", "jira", "agile", "scrum", "rest api",
    "graphql", "microservices", "unit testing", "test driven development",
    "object oriented programming", "oop", "design patterns",
    # Soft Skills
    "communication", "teamwork", "problem solving", "leadership",
    "project management", "critical thinking", "time management",
    "collaboration", "presentation", "mentoring",
    # Other
    "excel", "tableau", "power bi", "figma", "adobe", "photoshop",
    "cybersecurity", "networking", "blockchain", "hadoop", "spark",
]


def load_spacy_model():
    """Load the spaCy English model. Falls back gracefully if not installed."""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        # If model not found, use a blank English pipeline
        print("⚠️  spaCy model 'en_core_web_sm' not found. Using blank model.")
        nlp = spacy.blank("en")
        return nlp


def clean_text(text: str) -> str:
    """
    Clean and normalize text:
    - Lowercase everything
    - Remove special characters except spaces and dots
    - Collapse multiple spaces
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\+#]", " ", text)  # keep alphanumeric + common symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text: str) -> list:
    """
    Extract skills from a given text by matching against the SKILLS_DB list.

    Args:
        text (str): Raw text from a resume or job description.

    Returns:
        list: Sorted list of unique skills found in the text.
    """
    cleaned = clean_text(text)
    found_skills = set()

    # Match each skill in the database against the cleaned text
    for skill in SKILLS_DB:
        # Use word-boundary-aware matching to avoid partial word matches
        # e.g., "r" should not match inside "react" or "ruby"
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, cleaned):
            found_skills.add(skill.title())  # Store in Title Case for display

    return sorted(list(found_skills))


def extract_keywords_spacy(text: str, nlp) -> list:
    """
    Use spaCy to extract noun chunks and named entities as additional keywords.
    This supplements the skill list matching.

    Args:
        text (str): Raw text.
        nlp: Loaded spaCy model.

    Returns:
        list: List of extracted keyword strings.
    """
    doc = nlp(text[:100000])  # spaCy has a limit; truncate very long texts
    keywords = set()

    # Extract named entities (e.g., ORG, PRODUCT can be tools/technologies)
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT", "GPE", "WORK_OF_ART"):
            keywords.add(ent.text.lower())

    # Extract noun chunks (e.g., "machine learning", "data pipeline")
    for chunk in doc.noun_chunks:
        keywords.add(chunk.text.lower())

    return list(keywords)
