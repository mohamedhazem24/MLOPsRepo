import pandas as pd
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib
from loguru import logger
import os
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score
from sklearn.preprocessing import FunctionTransformer

# Setup logger
logger.add("training_log.log", rotation="500 KB", retention="5 days", level="INFO")
def preprocess_df(X,drop_cols):
    X = X.copy()
    X = X.drop(columns=drop_cols)
    X["Cabin"] = X["Cabin"].astype(str).str[0]
    return X
def preprocess_df2(X):
    X = X.copy()
    X = X.drop(columns=['PassengerId', 'Name', 'Ticket'])
    X["Cabin"] = X["Cabin"].astype(str).str[0]
    return X
# ... (your existing imports remain the same)

@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: DictConfig):
    logger.info("🧠 Starting model training with GridSearchCV...")
    paths=OmegaConf.create(cfg.paths)
    drop_cols = cfg.preprocessing.drop_cols
    cat_features = list(cfg.preprocessing.cat_features)
    num_features = list(cfg.preprocessing.num_features)
    passthrough_features = list(cfg.preprocessing.passthrough_features)
    # Convert paths to absolute paths
    features_path = Path(to_absolute_path(cfg.paths.processed_data_dir)) / "train_raw.csv"
    model_lr_path = Path(to_absolute_path(cfg.paths.models_dir)) / "logistic_regression_model.pkl"
    model_rf_path = Path(to_absolute_path(cfg.paths.models_dir)) / "random_forest_model.pkl"
    full_pipeline_path:Path = Path(to_absolute_path(cfg.paths.processed_data_dir))/"preprocessing_pipeline.pkl"
    model_lr_pipeline_path = Path(to_absolute_path(cfg.paths.models_dir)) / "logistic_regression_pipelined.pkl"
    model_rf_pipeline_path = Path(to_absolute_path(cfg.paths.models_dir)) / "random_forest_pipelined.pkl"
        
    
    cat_columns = [f"cat_{col}" for col in cat_features]
    num_columns = num_features.copy()
    all_columns = cat_columns + num_columns + passthrough_features
    # Load data
    
    with open(full_pipeline_path, "rb") as file:
        full_pipeline = joblib.load(file)

    logger.info(f"📂 Loading data from {features_path}...")
    dagshub.init(repo_owner='Mhazem', repo_name='my-first-repo', mlflow=True)
    df = pd.read_csv(features_path)

    # Apply preprocessing
    #df = pd.DataFrame(full_pipeline.transform(df), columns=all_columns)
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
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
    logger.error(type(lr_param_grid) )
    logger.error(lr_param_grid )
    # Create pipeline with the preprocessing and model
    lr_pipeline = Pipeline([
        ('preprocessor', full_pipeline),
        ('classifier', lr_model)
    ])
    
    lr_grid = GridSearchCV(
        lr_pipeline,
        lr_param_grid,
        cv=cfg.training.cv_folds,
        scoring=cfg.training.scoring_metric,
        n_jobs=-1
    )
    lr_grid.fit(X_train, y_train)
    
    logger.success(f"Best LR params: {lr_grid.best_params_}")
    logger.info(f"Best CV accuracy: {lr_grid.best_score_:.4f}")
    
    # Save the best pipeline (which already includes preprocessing)
    joblib.dump(lr_grid.best_estimator_, model_lr_path)
    joblib.dump(lr_grid.best_estimator_, model_lr_pipeline_path)
    
    with mlflow.start_run():
        logger.warning(lr_grid.best_estimator_)
        y_pred = lr_grid.best_estimator_.predict(X_test)
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        })
        mlflow.sklearn.log_model(lr_grid.best_estimator_, "Logistic_Regression_pipe")
        mlflow.log_artifact(model_lr_pipeline_path)

    # GridSearchCV for Random Forest
    logger.info("🌲 Tuning Random Forest...")
    rf_param_grid = OmegaConf.to_container(cfg.hyperparameters.random_forest, resolve=True)

    # Create pipeline with the preprocessing and model
    rf_pipeline = Pipeline([
        ('preprocessor', full_pipeline),
        ('classifier', rf_model)
    ])
    
    rf_grid = GridSearchCV(
        rf_pipeline,
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
    joblib.dump(rf_grid.best_estimator_, model_rf_path)
    joblib.dump(rf_grid.best_estimator_, model_rf_pipeline_path)
    
    with mlflow.start_run():
        y_pred = rf_grid.best_estimator_.predict(X_test)
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        })
        mlflow.sklearn.log_model(rf_grid.best_estimator_, "Random_Forest_pipe")
        mlflow.log_artifact(model_rf_pipeline_path)
        
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        try:
            client.create_registered_model("Random_Forest")
        except Exception:
            pass  # model already exists

        result = client.create_model_version(
            name="Random_Forest",
            source=model_uri,
            run_id=run_id
        )
        client.transition_model_version_stage(
            name="Random_Forest",
            version=result.version,
            stage="Production"
        )

    logger.success(f"Models saved to {cfg.paths.models_dir}")
    logger.success("🎉 Training complete!")

if __name__ == "__main__":
    main()