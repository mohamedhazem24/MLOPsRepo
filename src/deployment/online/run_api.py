import hydra
from omegaconf import DictConfig
from api import InferenceAPI  # Make sure this is the correct import path
import litserve as ls
def preprocess_df(X,drop_cols):
    X = X.copy()
    X = X.drop(columns=drop_cols)
    X["Cabin"] = X["Cabin"].astype(str).str[0]
    return X
import hydra
from omegaconf import DictConfig
from api import InferenceAPI  # Make sure this is the correct import path
import litserve as ls

@hydra.main(version_base=None, config_path="../../", config_name="config.yaml")
def main(cfg: DictConfig):
    # Initialize and set up the API with the Hydra configuration
    api = InferenceAPI()
    api.setup(cfg=cfg)
    #print(dir(ls))
    # Serve the API on port 8000 using ls.serve() (not LitServer)
    ls.server(api, port=8000)

if __name__ == "__main__":
    main()

