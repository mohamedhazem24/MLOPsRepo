from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer
from sklearn.pipeline import make_pipeline

import joblib
import pickle
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import json
import mlflow
import os
from dotenv import load_dotenv
import dagshub
import mlflow
import mlflow.sklearn  
from functools import partial
def preprocess_df(X,drop_cols):
    X = X.copy()
    X = X.drop(columns=drop_cols)
    X["Cabin"] = X["Cabin"].astype(str).str[0]
    return X
@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: DictConfig):
    logger.info("Starting evaluation pipeline...")

    try:
        features_path: Path = Path(cfg.paths.processed_data_dir) / 'test_raw.csv'
        model_path_1: Path = Path(to_absolute_path(cfg.paths.models_dir)) / "logistic_regression_model.pkl"
        model_path_2: Path = Path(to_absolute_path(cfg.paths.models_dir)) / "random_forest_model.pkl"
        pipeline_path: Path = Path(cfg.paths.processed_data_dir) /'preprocessing_pipeline.pkl'
        metrics_path: Path = Path(cfg.paths.reports_dir) / 'classification_metrics.json'
        model_lr_pipeline_path = Path(to_absolute_path(cfg.paths.models_dir)) / "logistic_regression_pipelined.pkl"
        model_rf_pipeline_path = Path(to_absolute_path(cfg.paths.models_dir)) / "random_forest_pipelined.pkl"
        load_dotenv()
        username = os.getenv("DAGSHUB_USERNAME")
        token = os.getenv("DAGSHUB_TOKEN")
        #mlflow.set_tracking_uri(f"https://dagshub.com/{username}/my-first-repo.mlflow")
        
        mlflow.set_tracking_uri(f"https://{username}:{token}@dagshub.com/Mhazem/my-first-repo.mlflow")

        #mlflow.set_tracking_uri("https://dagshub.com/Mhazem/my-first-repo")

        # Create reports directory if it doesn't exist
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Loading models from {model_path_1} and {model_path_2}")
        
        with open(model_lr_pipeline_path, 'rb') as f:
            model_1 = joblib.load(f)
        with open(model_rf_pipeline_path, 'rb') as f:
            model_2 = joblib.load(f)
        '''
        logged_model = 'runs:/609ef5faf94b42edbd8c3bc2347f9e7b/Random_Forest'

        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        
        print(f"The Type :{type(loaded_model)}")
        logger.error(loaded_model) 
        '''
        with open(pipeline_path,'rb') as f:
            pipeline=joblib.load(f)
        
        drop_cols = cfg.preprocessing.drop_cols
        cat_features = list(cfg.preprocessing.cat_features)
        num_features = list(cfg.preprocessing.num_features)
        passthrough_features = list(cfg.preprocessing.passthrough_features)
        test_size = cfg.preprocessing.test_size
        random_state = cfg.preprocessing.random_state

        partial_preprocess_df = partial(preprocess_df, drop_cols=drop_cols)

        # Load model as a PyFuncModel.
        #model_2 = mlflow.pyfunc.load_model(logged_model)
        cat_columns = [f"cat_{col}" for col in cat_features]
        num_columns = num_features.copy()
        all_columns = cat_columns + num_columns + passthrough_features
        df = pd.read_csv(features_path)
        X = df.drop("Survived", axis=1)

        
        y_pred=model_1.predict(X)
        #y_pred_trail=loaded_model.predict(X)
        #logger.warning("Predictionss")
        #logger.warning(y_pred_trail)
        
        
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
        model_registry = {
            "logistic_regression": {
                "path":model_path_1 ,
                "model": model_1
            },
            "random_forest": {
                "path":model_path_2 ,
                "model": model_2
            }
        }

        for metric_dict in all_metrics.values():
            with mlflow.start_run():
                mlflow.log_metrics({
                "accuracy": metric_dict['accuracy'],
                "precision": metric_dict['precision'],
                "recall": metric_dict['recall'],
                "f1_score": metric_dict['f1_score']
                })
                model_name=metric_dict['model']
                mlflow.sklearn.log_model(model_registry[model_name]['model'], model_name)
                mlflow.log_artifact(model_registry[model_name]['path'])
                
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