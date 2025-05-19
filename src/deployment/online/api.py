import numpy as np
import litserve as ls
import pickle
from functools import partial
from request import InferenceRequest
from hydra.utils import to_absolute_path
from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv
def preprocess_df(X,drop_cols):
    X = X.copy()
    X = X.drop(columns=drop_cols)
    X["Cabin"] = X["Cabin"].astype(str).str[0]
    return X
class InferenceAPI(ls.LitAPI):
    def setup(self, device = "cpu",cfg=None):
        self.features_path: Path = Path(cfg.paths.processed_data_dir) / 'test_raw.csv'
        self.model_path_1: Path = Path(to_absolute_path(cfg.paths.models_dir)) / "logistic_regression_model.pkl"
        self.pipeline_path: Path = Path(cfg.paths.processed_data_dir) /'preprocessing_pipeline.pkl'
        self.drop_cols = cfg.preprocessing.drop_cols
        self.cat_features = list(cfg.preprocessing.cat_features)
        self.num_features = list(cfg.preprocessing.num_features)
        self.passthrough_features = list(cfg.preprocessing.passthrough_features)
        self.cat_columns = [f"cat_{col}" for col in self.cat_features]
        self.num_columns = self.num_features.copy()
        self.all_columns = self.cat_columns + self.num_columns + self.passthrough_features

        load_dotenv()
        self.username = os.getenv("DAGSHUB_USERNAME")
        self.token = os.getenv("DAGSHUB_TOKEN")

        with open(cfg.paths.model_path_1, "rb") as pkl:
            self._model = pickle.load(pkl)
        with open("/teamspace/studios/this_studio/mlopsrepo/data/processed/preprocessing_pipeline.pkl", "rb") as pkl:
            self._pipeline = pickle.load(pkl)
        

    def decode_request(self, request):
        try:
            InferenceRequest(**request["input"])
            data = [(ky,val) for ky,val in request["input"]]
            x=pd.DataFrame(data)
            x=pd.DataFrame(self._pipeline.transform(x))
            x=x.drop('Survived',axis=1)
            return x
        except:
            return None

    def predict(self, x):
        if x is not None:
            return self._model.predict(x)
        else:
            return None

    def encode_response(self, output):
        if output is None:
            message = "Error Occurred"
        else:
            message = "Response Produced Successfully"
        return {
            "message": message,
            "prediction": [output]
        }