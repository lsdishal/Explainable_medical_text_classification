import pytest
import pandas as pd
from app.components.header import render_header
from app.components.footer import render_footer

def test_frontend_component_imports():
    from app.components.sidebar import render_sidebar
    from app.components.cards import render_prediction_card
    from app.components.explanation_tabs import render_explanation_tabs
    assert callable(render_sidebar)
    assert callable(render_prediction_card)
    assert callable(render_explanation_tabs)

def test_preset_example_data():
    import importlib
    pred_module = importlib.import_module("app.pages.2_Prediction")
    presets = getattr(pred_module, "PRESETS", {})
    assert "Migraine" in presets
    assert len(presets) >= 3
    for label, text in presets.items():
        assert len(text) > 10
