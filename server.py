from src.deployment.online import api
import litserve as ls

def preprocess_df(X,drop_cols):
    X = X.copy()
    X = X.drop(columns=drop_cols)
    X["Cabin"] = X["Cabin"].astype(str).str[0]
    return X

if __name__ == "__main__":
    api = api.InferenceAPI()
    server = ls.LitServer(
        api, 
        accelerator="cpu"
    )
    server.run(port=8000, generate_client_file = False)
