import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import joblib
from omegaconf import OmegaConf

from loguru import logger
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

# Setup logger
logger.add("training_log.log", rotation="500 KB", retention="5 days", level="INFO")

@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: DictConfig):
    logger.info("🧠 Starting model training with GridSearchCV...")
    
    # Convert paths to absolute paths
    features_path = Path(to_absolute_path(cfg.paths.processed_data_dir)) / "train.csv"
    model_lr_path = Path(to_absolute_path(cfg.paths.models_dir)) / "logistic_regression_model.pkl"
    model_rf_path = Path(to_absolute_path(cfg.paths.models_dir)) / "random_forest_model.pkl"

    # Load data
    logger.info(f"📂 Loading data from {features_path}...")
    df = pd.read_csv(features_path)
    X = df.drop(cfg.training.target_column, axis=1)
    y = df[cfg.training.target_column]
    logger.success(f"✅ Data loaded. Features shape: {X.shape}")

    # Train-test split
    logger.info(f"✂️ Splitting data (test_size={cfg.training.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state
    )
    logger.success(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Initialize models
    logger.info("🛠️ Initializing models...")
    lr_model = LogisticRegression(
        max_iter=cfg.models.logistic_regression.max_iter,
        random_state=cfg.training.random_state
    )
    rf_model = RandomForestClassifier(random_state=cfg.training.random_state)

    # GridSearchCV for Logistic Regression
    logger.info("🔍 Tuning Logistic Regression...")
    lr_param_grid = OmegaConf.to_container(cfg.hyperparameters.logistic_regression, resolve=True)
    
    lr_grid = GridSearchCV(
        lr_model,
        lr_param_grid,
        cv=cfg.training.cv_folds,
        scoring=cfg.training.scoring_metric,
        n_jobs=-1
    )
    lr_grid.fit(X_train, y_train)
    logger.success(f"Best LR params: {lr_grid.best_params_}")
    logger.info(f"Best CV accuracy: {lr_grid.best_score_:.4f}")

    # GridSearchCV for Random Forest
    logger.info("🌲 Tuning Random Forest...")
    rf_param_grid = OmegaConf.to_container(cfg.hyperparameters.random_forest, resolve=True)

    rf_grid = GridSearchCV(
        rf_model,
        rf_param_grid,
        cv=cfg.training.cv_folds,
        scoring=cfg.training.scoring_metric,
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    logger.success(f"Best RF params: {rf_grid.best_params_}")
    logger.info(f"Best CV accuracy: {rf_grid.best_score_:.4f}")

    # Save models
    logger.info("💾 Saving models...")
    joblib.dump(lr_grid.best_estimator_, model_lr_path)
    joblib.dump(rf_grid.best_estimator_, model_rf_path)
    logger.success(f"Models saved to {cfg.paths.models_dir}")

    logger.success("🎉 Training complete!")

if __name__ == "__main__":
    main()