import pytest
import pandas as pd
from src.preprocessing.cleaner import clean_medical_text

def test_inference_text_cleaning():
    sample_text = "I have a cough and high fever for 3 days."
    processed = clean_medical_text(sample_text)
    assert len(processed) > 0
    assert isinstance(processed, str)
