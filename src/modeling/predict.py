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
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: DictConfig):
    logger.info("Starting evaluation pipeline...")

    try:
        features_path: Path = Path(cfg.paths.processed_data_dir) / 'test.csv'
        model_path_1: Path = Path(to_absolute_path(cfg.paths.models_dir)) / "logistic_regression_model.pkl"
        model_path_2: Path = Path(to_absolute_path(cfg.paths.models_dir)) / "random_forest_model.pkl"
        predictions_path_1: Path = Path(cfg.paths.processed_data_dir) / 'pred1.csv'
        predictions_path_2: Path = Path(cfg.paths.processed_data_dir) / 'pred2.csv'

        logger.debug(f"Loading models from {model_path_1} and {model_path_2}")
        with open(model_path_1, 'rb') as f:
            model_1 = joblib.load(f)
        with open(model_path_2, 'rb') as f:
            model_2 = joblib.load(f)

        logger.debug(f"Reading test features from {features_path}")
        df = pd.read_csv(features_path)
        X = df.drop("Survived", axis=1)
        y = df[['Survived']]

        logger.info("Generating predictions...")
        y_pred_1 = model_1.predict(X)
        y_pred_2 = model_2.predict(X)

        logger.info(f"Report for Model 1 ({model_1}):")
        logger.info("\n" + classification_report(y, y_pred_1))

        logger.info(f"Report for Model 2 ({model_2}):")
        logger.info("\n" + classification_report(y, y_pred_2))

    except Exception as e:
        logger.exception(f"An error occurred during evaluation: {e}")


if __name__ == "__main__":
    main()
