import pytest
from src.preprocessing.cleaner import clean_medical_text

def test_clean_medical_text():
    dirty_text = "  High fever,\r\n\t shivering and   chills!!  "
    cleaned = clean_medical_text(dirty_text)
    assert cleaned == "High fever, shivering and chills!!"
    assert "  " not in cleaned
