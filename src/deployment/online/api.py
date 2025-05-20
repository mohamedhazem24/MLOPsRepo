import numpy as np
import litserve as ls
import pickle
from functools import partial
from .request import InferenceRequest
from hydra.utils import to_absolute_path
from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import litserve as ls
import pickle
from fastapi import FastAPI
import joblib
import uvicorn


def preprocess_df(X,drop_cols):
    X = X.copy()
    X = X.drop(columns=drop_cols)
    
    X["Cabin"] = X["Cabin"].astype(str).str[0]
    return X
class InferenceAPI(ls.LitAPI):
    def setup(self, device = "cpu"):
        with open("/teamspace/studios/this_studio/mlopsrepo/models/logistic_regression_pipelined.pkl", "rb") as pkl:
            self._model = joblib.load(pkl)
        self._encoder= {1:"Survived",0:"Died"}
    def decode_request(self, request):
        try:
            InferenceRequest(**request["input"])
            data = request["input"]
            x = pd.DataFrame([data])
            print(x)
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
            "prediction": [self._encoder[val] for val in output]
        }

