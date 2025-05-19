import pandas as pd
from pathlib import Path
import joblib
import typer
from loguru import logger
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

logger.add("predict_log.log", rotation="500 KB", retention="5 days", level="INFO")
def preprocess_df(X,drop_cols):
    logger.debug("🔍 Extracting first letter from 'Cabin' and dropping unused columns...")
    X = X.copy()
    X = X.drop(columns=drop_cols)
    X["Cabin"] = X["Cabin"].astype(str).str[0]
    return X
@hydra.main(version_base=None, config_path="../../", config_name="config.yaml")
def main(cfg: DictConfig):
    logger.info("🔍 Starting prediction pipeline...")

    # Load raw data
    raw_path = Path(to_absolute_path(cfg.paths.raw_data_dir)) / "titanic.csv"
    df = pd.read_csv(raw_path)
    logger.success(f"📂 Raw data loaded. Shape: {df.shape}")

    # Prepare preprocessing pipeline
    drop_cols = cfg.preprocessing.drop_cols
    cat_features = list(cfg.preprocessing.cat_features)
    num_features = list(cfg.preprocessing.num_features)
    passthrough_features = list(cfg.preprocessing.passthrough_features)

    def preprocess_df(X):
        X = X.copy()
        X = X.drop(columns=drop_cols)
        X["Cabin"] = X["Cabin"].astype(str).str[0]
        return X
    pipeline_path = Path(to_absolute_path(cfg.paths.processed_data_dir)) / "preprocessing_pipeline.pkl"
    logger.error(pipeline_path)
    with open(pipeline_path ,'rb') as f:
        pipeline = joblib.load(f)
    logger.success("🧱 Loaded preprocessing pipeline")
    
    logger.info("🛠️ Applying preprocessing pipeline...")
    processed = pipeline.fit_transform(df)
        # Get column names
    cat_columns = [f"cat_{col}" for col in cat_features]
    num_columns = num_features.copy()
    all_columns = cat_columns + num_columns + passthrough_features
    
    # Create DataFrames
    processed = pd.DataFrame(processed, columns=all_columns)

    processed = processed.drop('Survived',axis=1)
    # Load trained models
    model_path_1 = Path(to_absolute_path(cfg.paths.models_dir)) / "logistic_regression_model.pkl"
    model_path_2 = Path(to_absolute_path(cfg.paths.models_dir)) / "random_forest_model.pkl"
    
    model_1 = joblib.load(model_path_1)
    model_2 = joblib.load(model_path_2)
    logger.success("✅ Models loaded")

    # Make predictions
    pred_1 = model_1.predict(processed)
    pred_2 = model_2.predict(processed)

    # Add predictions to dataframe
    df['prediction_logistic'] = pred_1
    df['prediction_rf'] = pred_2

    # Save predictions
    output_path = Path(to_absolute_path(cfg.paths.processed_data_dir)) / "predictions.csv"
    df.to_csv(output_path, index=False)
    logger.success(f"🎯 Predictions saved to: {output_path}")

if __name__ == "__main__":
    main()
