# utils.py
# Utility functions for PDF text extraction and text preprocessing.

import pdfplumber
import io


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract all text from an uploaded PDF file using pdfplumber.

    Args:
        uploaded_file: A file-like object (e.g., from Streamlit's file_uploader).

    Returns:
        str: All extracted text from the PDF, or an empty string on failure.
    """
    text = ""

    try:
        # Read the uploaded file bytes into an in-memory buffer
        pdf_bytes = uploaded_file.read()
        pdf_buffer = io.BytesIO(pdf_bytes)

        # Open the PDF with pdfplumber
        with pdfplumber.open(pdf_buffer) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

    return text.strip()


def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing:
    - Strip leading/trailing whitespace
    - Replace multiple newlines with a single space
    - Remove non-printable characters

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned text.
    """
    import re

    if not text:
        return ""

    # Replace newlines and tabs with spaces
    text = re.sub(r"[\n\t\r]+", " ", text)

    # Remove non-ASCII characters (optional: keeps text clean)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r" +", " ", text)

    return text.strip()


def truncate_text(text: str, max_chars: int = 5000) -> str:
    """
    Truncate text to a maximum number of characters for display purposes.

    Args:
        text (str): Input text.
        max_chars (int): Maximum characters to return.

    Returns:
        str: Truncated text with an ellipsis if it was cut.
    """
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def format_skill_list_as_tags(skills: list) -> str:
    """
    Format a list of skills into a comma-separated string.

    Args:
        skills (list): List of skill strings.

    Returns:
        str: Comma-separated skills string, or 'None found' if empty.
    """
    if not skills:
        return "None found"
    return ", ".join(skills)
