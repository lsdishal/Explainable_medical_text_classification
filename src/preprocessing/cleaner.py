import re

def clean_medical_text(text: str) -> str:
    """Normalizes medical symptom description text."""
    if not isinstance(text, str):
        return ""

    # Remove unwanted non-printable characters
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Standardize multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text
