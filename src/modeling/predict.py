from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer
import joblib
import pickle
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import json

@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: DictConfig):
    logger.info("Starting evaluation pipeline...")

    try:
        features_path: Path = Path(cfg.paths.processed_data_dir) / 'test.csv'
        model_path_1: Path = Path(to_absolute_path(cfg.paths.models_dir)) / "logistic_regression_model.pkl"
        model_path_2: Path = Path(to_absolute_path(cfg.paths.models_dir)) / "random_forest_model.pkl"
        metrics_path: Path = Path(cfg.paths.reports_dir) / 'classification_metrics.json'

        # Create reports directory if it doesn't exist
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Loading models from {model_path_1} and {model_path_2}")
        with open(model_path_1, 'rb') as f:
            model_1 = joblib.load(f)
        with open(model_path_2, 'rb') as f:
            model_2 = joblib.load(f)

        logger.debug(f"Reading test features from {features_path}")
        df = pd.read_csv(features_path)
        X = df.drop("Survived", axis=1)
        y = df['Survived']  # Removed double brackets to make it a Series

        logger.info("Generating predictions...")
        y_pred_1 = model_1.predict(X)
        y_pred_2 = model_2.predict(X)

        def calculate_metrics(y_true, y_pred, model_name):
            """Calculate classification metrics and return as dictionary"""
            metrics = {
                "model": model_name,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted'),
                "recall": recall_score(y_true, y_pred, average='weighted'),
                "f1_score": f1_score(y_true, y_pred, average='weighted'),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
                "classification_report": classification_report(y_true, y_pred, output_dict=True)
            }
            return metrics

        # Calculate metrics for both models
        metrics_1 = calculate_metrics(y, y_pred_1, "logistic_regression")
        metrics_2 = calculate_metrics(y, y_pred_2, "random_forest")

        # Combine metrics into one dictionary
        all_metrics = {
            "model_1": metrics_1,
            "model_2": metrics_2
        }

        # Save metrics to JSON file
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)

        logger.info(f"Metrics saved to {metrics_path}")

        # Print reports to console
        logger.info(f"Report for Model 1 ({model_1}):")
        logger.info("\n" + classification_report(y, y_pred_1))

        logger.info(f"Report for Model 2 ({model_2}):")
        logger.info("\n" + classification_report(y, y_pred_2))

    except Exception as e:
        logger.exception(f"An error occurred during evaluation: {e}")


if __name__ == "__main__":
    main()