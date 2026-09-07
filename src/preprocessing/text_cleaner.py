import re
import html
from typing import Dict, Any, Tuple

# Essential negation words to strictly preserve in medical context
NEGATION_WORDS = {
    "no", "not", "without", "never", "neither", "nor", "none",
    "lack", "lacks", "lacking", "absent", "non", "cannot", "n't"
}

def clean_medical_text(text: str) -> str:
    """Normalizes medical text while preserving clinical meaning, punctuation, medical terms, and negations.
    
    Strict Rules:
    - NO stopword removal
    - NO stemming or lemmatization
    - Preservation of negations ('no', 'not', 'without', etc.)
    - Removal of HTML tags and unescaping entities
    - Whitespace normalization
    """
    if not isinstance(text, str):
        return ""

    # 1. Unescape HTML entities (e.g. &amp; -> &, &lt; -> <)
    text = html.unescape(text)

    # 2. Strip HTML/markup tags if present
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Replace non-breaking space and special unicode spaces with standard space
    text = text.replace("\xa0", " ").replace("\u200b", "")

    # 4. Standardize hyphens/dashes in compound medical words (e.g., 'short-term' -> 'short-term')
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", text)

    # 5. Normalize whitespace (tabs, newlines, multiple spaces)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    # 6. Strip leading and trailing whitespace
    text = text.strip()

    return text

def calculate_text_stats(text: str) -> Dict[str, int]:
    """Calculates character count and word count statistics for a single text record."""
    clean_t = clean_medical_text(text)
    words = clean_t.split()
    return {
        "char_length": len(clean_t),
        "word_count": len(words)
    }

def is_outlier_length(text: str, min_chars: int = 10, min_words: int = 3, max_chars: int = 1000) -> Tuple[bool, str]:
    """Detects whether a text record has an unusually short or unusually long length."""
    stats = calculate_text_stats(text)
    char_len = stats["char_length"]
    word_cnt = stats["word_count"]

    if char_len < min_chars or word_cnt < min_words:
        return True, "unusually_short"
    if char_len > max_chars:
        return True, "unusually_long"

    return False, "normal"

if __name__ == "__main__":
    raw_sample = "Patient reports &amp; complains of <br> severe fever without cough... No headache present!   "
    cleaned = clean_medical_text(raw_sample)
    print("Raw sample:    ", repr(raw_sample))
    print("Cleaned sample:", repr(cleaned))
    print("Stats:         ", calculate_text_stats(cleaned))
    print("Outlier check: ", is_outlier_length(cleaned))
