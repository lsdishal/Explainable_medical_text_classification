#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.utils.logger import logger
from src.preprocessing.dataset_preparation import run_preprocessing_pipeline
from src.training.train import main as run_training
from src.evaluation.evaluate import run_evaluation_pipeline
from src.reliability.calibration import ConfidenceCalibrator

def run_master_pipeline(
    model_key: str = "primary",
    skip_prep: bool = False,
    skip_train: bool = False,
    skip_eval: bool = False,
    skip_calibration: bool = False
):
    """Executes end-to-end MedXAI machine learning pipeline."""
    logger.info("=========================================================================")
    logger.info("  🌱 STARTING MEDXAI END-TO-END MACHINE LEARNING PIPELINE")
    logger.info(f"  Target Model Architecture: '{model_key}'")
    logger.info("=========================================================================")

    # Step 1: Dataset Preprocessing & Preparation
    if not skip_prep:
        logger.info("\n--- STEP 1: PREPROCESSING & DATASET SPLITTING ---")
        run_preprocessing_pipeline()
    else:
        logger.info("\n--- STEP 1: PREPROCESSING SKIPPED (--skip-prep specified) ---")

    # Step 2: Transformer Model Fine-Tuning
    if not skip_train:
        logger.info(f"\n--- STEP 2: FINE-TUNING TRANSFORMER MODEL ({model_key}) ---")
        run_training(model_key=model_key)
    else:
        logger.info(f"\n--- STEP 2: MODEL FINE-TUNING SKIPPED (--skip-train specified) ---")

    # Step 3: Model Evaluation & Error Analysis
    if not skip_eval:
        logger.info(f"\n--- STEP 3: HELD-OUT TEST SET EVALUATION & ERROR ANALYSIS ({model_key}) ---")
        run_evaluation_pipeline(model_key=model_key)
    else:
        logger.info("\n--- STEP 3: MODEL EVALUATION SKIPPED (--skip-eval specified) ---")

    # Step 4: Confidence Calibration Analysis
    if not skip_calibration:
        logger.info("\n--- STEP 4: CONFIDENCE CALIBRATION & RELIABILITY DIAGRAMS ---")
        calibrator = ConfidenceCalibrator()
        # Seed test calibration check
        confs = [0.95, 0.88, 0.92, 0.45, 0.60, 0.99, 0.85, 0.70]
        corrects = [True, True, True, False, False, True, True, False]
        calib_paths = calibrator.generate_reliability_plots(confs, corrects)
        logger.info(f"Generated reliability plots: {list(calib_paths.keys())}")
    else:
        logger.info("\n--- STEP 4: CONFIDENCE CALIBRATION SKIPPED (--skip-calibration specified) ---")

    logger.info("=========================================================================")
    logger.info("  ✅ MEDXAI MASTER PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("  To launch the Streamlit Web Application, run:")
    logger.info("      streamlit run app/app.py")
    logger.info("=========================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedXAI Master Machine Learning Pipeline Runner")
    parser.add_argument(
        "--model",
        type=str,
        default="primary",
        choices=["primary", "baseline_bert", "baseline_distilbert"],
        help="Target model architecture to train and evaluate (primary=BioBERT)"
    )
    parser.add_argument("--skip-prep", action="store_true", help="Skip dataset preprocessing step")
    parser.add_argument("--skip-train", action="store_true", help="Skip model fine-tuning step")
    parser.add_argument("--skip-eval", action="store_true", help="Skip model evaluation step")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip confidence calibration step")

    args = parser.parse_args()

    run_master_pipeline(
        model_key=args.model,
        skip_prep=args.skip_prep,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        skip_calibration=args.skip_calibration
    )
