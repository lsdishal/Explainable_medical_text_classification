import pytest
import importlib

def test_multipage_imports():
    p1 = importlib.import_module("app.pages.1_Home")
    p2 = importlib.import_module("app.pages.2_Prediction")
    p3 = importlib.import_module("app.pages.3_Model_Evaluation")
    p4 = importlib.import_module("app.pages.4_Dataset_Analysis")
    p5 = importlib.import_module("app.pages.5_Error_Analysis")
    p6 = importlib.import_module("app.pages.6_About")

    assert p1 is not None
    assert p2 is not None
    assert p3 is not None
    assert p4 is not None
    assert p5 is not None
    assert p6 is not None
