import pandas as pd
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
from loguru import logger
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from pathlib import Path
# Setup logger
def preprocess_df(X,drop_cols):
    X = X.copy()
    X = X.drop(columns=drop_cols)
    X["Cabin"] = X["Cabin"].astype(str).str[0]
    return X
logger.add("pipeline_log.log", rotation="500 KB", retention="5 days", level="INFO")
@hydra.main(version_base=None, config_path='.',config_name="config.yaml")
def main(cfg: DictConfig):
    logger.info("🚀 Starting preprocessing pipeline...")
    
    # Convert paths to absolute paths
    input_path = Path(to_absolute_path(cfg.paths.raw_data_dir)) / "titanic.csv"
    output_path = Path(to_absolute_path(cfg.paths.processed_data_dir))
    
    logger.info(f"📂 Reading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    logger.success(f"✅ Dataset loaded. Shape: {df.shape}")

    # --- Get config values ---
    drop_cols = cfg.preprocessing.drop_cols
    cat_features = list(cfg.preprocessing.cat_features)
    num_features = list(cfg.preprocessing.num_features)
    passthrough_features = list(cfg.preprocessing.passthrough_features)
    test_size = cfg.preprocessing.test_size
    random_state = cfg.preprocessing.random_state

    # 💥 Split first
    logger.info(f"🔀 Splitting into train and test sets ({1-test_size:.0%}/{test_size:.0%})...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    train_df.to_csv(output_path / "train_raw.csv", index=False)
    test_df.to_csv(output_path / "test_raw.csv", index=False)
    logger.success(f"📊 Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Preprocessing function partial
    partial_preprocess_df = partial(preprocess_df, drop_cols=drop_cols)

    # 🧱 Build preprocessing pipeline
    logger.info("🧱 Building preprocessing pipeline...")
    full_pipeline = Pipeline([
        ("custom_preprocessing", FunctionTransformer(partial_preprocess_df)),
        ("preprocessing", ColumnTransformer([
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy=cfg.imputation.cat_strategy)),
                ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))
            ]), cat_features),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy=cfg.imputation.num_strategy)),
            ]), num_features),
            ("pass", "passthrough", passthrough_features)
        ], remainder="drop"))
    ])

    # 🧠 Fit-transform on train, transform on test
    logger.info("⚙️ Applying pipeline...")
    logger.error('Survived' in train_df.columns)
    X_train=train_df.drop('Survived',axis=1)
    train_array = full_pipeline.fit_transform(X_train)
    X_test=test_df.drop('Survived',axis=1)
    test_array = full_pipeline.transform(X_test)

    # Get column names
    cat_columns = [f"cat_{col}" for col in cat_features]
    num_columns = num_features.copy()
    all_columns = cat_columns + num_columns + passthrough_features
    
    # Create DataFrames
    train_processed = pd.DataFrame(train_array, columns=all_columns)
    test_processed = pd.DataFrame(test_array, columns=all_columns)

    # 💾 Save pipeline and datasets
    pipeline_path = output_path / "preprocessing_pipeline.pkl"
    joblib.dump(full_pipeline, pipeline_path)
    pipe_2=joblib.load(pipeline_path)
    print(pipe_2)
    logger.success(f"🧠 Preprocessing pipeline saved to {pipeline_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    train_processed.to_csv(output_path / "train_preprocessed.csv", index=False)
    test_processed.to_csv(output_path / "test_preprocessed.csv", index=False)
    logger.success("🎉 Preprocessing complete. Files saved!")
if __name__ == "__main__":
    main()