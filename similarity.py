# similarity.py
# This module computes similarity between resume and job description,
# and identifies matching vs. missing skills.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_match_percentage(resume_text: str, jd_text: str) -> float:
    """
    Calculate how similar a resume is to a job description using TF-IDF
    and cosine similarity.

    Args:
        resume_text (str): Full text from the resume.
        jd_text (str): Full text from the job description.

    Returns:
        float: Match percentage between 0 and 100.
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    # TF-IDF converts text into numerical vectors
    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        # Fit on both documents and transform them
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])

        # Cosine similarity gives a value between 0 (no match) and 1 (identical)
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

        # Convert to percentage and round to 2 decimal places
        percentage = round(float(similarity_score[0][0]) * 100, 2)
        return percentage

    except Exception as e:
        print(f"Error computing similarity: {e}")
        return 0.0


def get_matching_skills(resume_skills: list, jd_skills: list) -> list:
    """
    Find skills that appear in BOTH the resume and job description.

    Args:
        resume_skills (list): Skills extracted from the resume.
        jd_skills (list): Skills extracted from the job description.

    Returns:
        list: Skills found in both.
    """
    # Normalize to lowercase for comparison, then return Title Case
    resume_set = {skill.lower() for skill in resume_skills}
    jd_set = {skill.lower() for skill in jd_skills}

    matching = resume_set.intersection(jd_set)
    return sorted([skill.title() for skill in matching])


def get_missing_skills(resume_skills: list, jd_skills: list) -> list:
    """
    Find skills required by the job description that are MISSING from the resume.

    Args:
        resume_skills (list): Skills extracted from the resume.
        jd_skills (list): Skills extracted from the job description.

    Returns:
        list: Skills in the JD but not in the resume.
    """
    resume_set = {skill.lower() for skill in resume_skills}
    jd_set = {skill.lower() for skill in jd_skills}

    missing = jd_set.difference(resume_set)
    return sorted([skill.title() for skill in missing])


def get_skill_gap_summary(resume_skills: list, jd_skills: list) -> dict:
    """
    Generate a complete skill gap analysis summary.

    Args:
        resume_skills (list): Skills from the resume.
        jd_skills (list): Skills from the job description.

    Returns:
        dict: Summary with matching, missing, and coverage percentage.
    """
    matching = get_matching_skills(resume_skills, jd_skills)
    missing = get_missing_skills(resume_skills, jd_skills)

    total_jd_skills = len(jd_skills)
    coverage = (len(matching) / total_jd_skills * 100) if total_jd_skills > 0 else 0

    return {
        "matching_skills": matching,
        "missing_skills": missing,
        "total_resume_skills": len(resume_skills),
        "total_jd_skills": total_jd_skills,
        "matched_count": len(matching),
        "missing_count": len(missing),
        "skill_coverage_percentage": round(coverage, 2),
    }
