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
import numpy as np
import litserve as ls
import pickle
from fastapi import FastAPI
from lightning.app import LightningWork

import uvicorn


class InferenceAPI(ls.LitAPI):
    def setup(self, device = "cpu"):
        with open("/teamspace/studios/this_studio/mlopsrepo/models/logistic_regression_pipelined.pkl", "rb") as pkl:
            self._model = pickle.load(pkl)
        self._encoder= {1:"Survived",0:"Died"}
    def decode_request(self, request):
        try:
            InferenceRequest(**request["input"])
            data = [val for val in request["input"].values()]
            x = np.asarray(data)
            x = np.expand_dims(x, 0)
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

class ModelServer(LightningWork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api = InferenceAPI()
        self.server = ls.LitServer(self.api, port=self.port)

    def run(self):
        self.server.run()

# Create a Lightning App
app = FastAPI()
model_server = ModelServer(port=8000)

@app.post("/predict")
async def predict(request_data: dict):
    x = model_server.api.decode_request(request_data)
    output = model_server.api.predict(x)
    return model_server.api.encode_response(output)

# Run the app (for testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)