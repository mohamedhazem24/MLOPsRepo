import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pathlib import Path

from loguru import logger
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

# Setup logger
logger.add("pipeline_log.log", rotation="500 KB", retention="5 days", level="INFO")

@hydra.main(version_base=None, config_path='/teamspace/studios/this_studio/mlopsrepo/src/',config_name="config.yaml")
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
    cat_features = ["Sex", "Cabin", "Embarked"]
    num_features = ['Age']
    passthrough_features = ["Survived", "Pclass", "SibSp", "Parch", "Fare"]
    test_size = cfg.preprocessing.test_size
    random_state = cfg.preprocessing.random_state
    if ["Sex", "Cabin", "Embarked"] == cfg.preprocessing.cat_features:
        print("HeLL Yeaaaaaah!!!!!!")

    def preprocess_df(X):
        logger.debug("🔍 Extracting first letter from 'Cabin' and dropping unused columns...")
        X = X.copy()
        X = X.drop(columns=drop_cols)
        X["Cabin"] = X["Cabin"].astype(str).str[0]
        return X
    logger.info("🧱 Building preprocessing pipeline...")
    full_pipeline = Pipeline([
        ("custom_preprocessing", FunctionTransformer(preprocess_df)),
        ("preprocessing", ColumnTransformer([
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy=cfg.imputation.cat_strategy)),
                ("encoder", OrdinalEncoder())
            ]), cat_features),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy=cfg.imputation.num_strategy)),
            ]), num_features),
            ("pass", "passthrough", passthrough_features)
        ], remainder="drop"))  # Changed from "passthrough" to "drop" to be explicit
    ])

    logger.info("⚙️ Transforming dataset with pipeline...")
    processed_array = full_pipeline.fit_transform(df)
    
    # Get feature names after transformation
    cat_transformer = full_pipeline.named_steps['preprocessing'].named_transformers_['cat']
    num_transformer = full_pipeline.named_steps['preprocessing'].named_transformers_['num']
    
    # Create column names for the transformed features
    cat_columns = [f"cat_{col}" for col in cat_features]
    num_columns = num_features.copy()
    all_columns = cat_columns + num_columns + passthrough_features
    
    processed_df = pd.DataFrame(processed_array, columns=all_columns)
    logger.success(f"✅ Transformation complete. Shape: {processed_df.shape}")

    logger.info(f"🔀 Splitting into train and test sets ({1-test_size:.0%}/{test_size:.0%})...")
    train, test = train_test_split(
        processed_df, 
        test_size=test_size, 
        random_state=random_state
    )
    logger.success(f"📊 Train shape: {train.shape}, Test shape: {test.shape}")

    logger.info(f"💾 Saving processed data to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path / "train.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)
    logger.success("🎉 Preprocessing complete. Files saved!")

if __name__ == "__main__":
    main()