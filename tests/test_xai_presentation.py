import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.inference.predict import MedicalSymptomClassifier
from src.explainability.explanation_manager import ExplanationManager
from src.explainability.subword_aggregator import aggregate_subwords

def test_xai_presentation():
    print("=== Testing XAI Subword Aggregation & Presentation Layer ===")

    # 1. Test Subword Aggregator Unit Logic
    raw_sample = [
        ("[CLS]", 0.0),
        ("i", 0.01),
        ("have", 0.02),
        ("se", 0.05),
        ("##vere", 0.15),
        ("head", 0.20),
        ("##ache", 0.40),
        ("with", 0.05),
        ("itch", 0.10),
        ("##ing", 0.25),
        (".", 0.0),
        ("[SEP]", 0.0)
    ]
    aggregated = aggregate_subwords(raw_sample)
    print("Unit Aggregation Output:", aggregated)

    agg_dict = dict(aggregated)
    assert "severe" in agg_dict, "Failed to aggregate 'se' + '##vere' -> 'severe'"
    assert abs(agg_dict["severe"] - 0.20) < 1e-5, f"Expected 0.20 for severe, got {agg_dict['severe']}"

    assert "headache" in agg_dict, "Failed to aggregate 'head' + '##ache' -> 'headache'"
    assert abs(agg_dict["headache"] - 0.60) < 1e-5, f"Expected 0.60 for headache, got {agg_dict['headache']}"

    assert "itching" in agg_dict, "Failed to aggregate 'itch' + '##ing' -> 'itching'"
    assert abs(agg_dict["itching"] - 0.35) < 1e-5, f"Expected 0.35 for itching, got {agg_dict['itching']}"

    assert "[CLS]" not in agg_dict and "[SEP]" not in agg_dict, "Special tokens not filtered out!"
    print("✓ Subword Aggregator Unit Tests Passed!")

    # 2. Test Full Live Model Explanation Manager Integration
    clf = MedicalSymptomClassifier()
    text = "I have a severe headache with nausea and sensitivity to light."
    pred_res = clf.predict(text)
    top_pred = pred_res["top_3_predictions"][0]

    explainer = ExplanationManager(clf.model, clf.tokenizer, clf.label_list, clf.device)
    xai_out = explainer.explain(text, top_pred["class_id"], top_pred["disease"], top_pred["confidence"], use_cache=False)

    print("\n--- Live XAI Manager Validation ---")
    print("Prediction:", xai_out["prediction"])
    print("Confidence:", xai_out["confidence"])

    # Verify all 4 methods generated output
    assert "attention_importance" in xai_out, "Attention missing!"
    assert "shap_importance" in xai_out, "SHAP missing!"
    assert "lime_importance" in xai_out, "LIME missing!"
    assert "integrated_gradients_importance" in xai_out, "Integrated Gradients missing!"

    # Verify RAW token scores preserved
    att_raw = xai_out["attention_importance"].get("raw_token_scores")
    shap_raw = xai_out["shap_importance"].get("raw_token_scores")
    lime_raw = xai_out["lime_importance"].get("raw_token_scores")
    ig_raw = xai_out["integrated_gradients_importance"].get("raw_token_scores")

    assert att_raw is not None and len(att_raw) > 0, "RAW Attention token scores missing!"
    assert shap_raw is not None and len(shap_raw) > 0, "RAW SHAP token scores missing!"
    assert lime_raw is not None and len(lime_raw) > 0, "RAW LIME token scores missing!"
    assert ig_raw is not None and len(ig_raw) > 0, "RAW Integrated Gradients token scores missing!"

    print("✓ Raw token attributions preserved for all 4 methods!")

    # Verify HUMAN-READABLE word attributions generated
    att_words = xai_out["attention_importance"].get("word_attributions")
    shap_words = xai_out["shap_importance"].get("word_attributions")
    lime_words = xai_out["lime_importance"].get("word_attributions")
    ig_words = xai_out["integrated_gradients_importance"].get("word_attributions")

    assert att_words is not None and len(att_words) > 0, "Human-readable Attention words missing!"
    assert shap_words is not None and len(shap_words) > 0, "Human-readable SHAP words missing!"
    assert lime_words is not None and len(lime_words) > 0, "Human-readable LIME words missing!"
    assert ig_words is not None and len(ig_words) > 0, "Human-readable IG words missing!"

    print("Human-Readable Attention Words:", att_words[:5])
    print("Human-Readable SHAP Words:", shap_words[:5])
    print("Human-Readable LIME Words:", lime_words[:5])
    print("Human-Readable Integrated Gradients Words:", ig_words[:5])

    # Verify Explanation Agreement works on normalized words
    consistent = xai_out.get("consistent_influential_features", [])
    print("Consistently Influential Human-Readable Features:", [f["word"] for f in consistent])

    assert len(xai_out["summary_table"]) > 0, "Summary table missing!"
    print("✓ Explanation Agreement Table generated successfully on human-readable words!")
    print("\n🎉 ALL XAI PRESENTATION TESTS PASSED CLEANLY!")

if __name__ == "__main__":
    test_xai_presentation()
