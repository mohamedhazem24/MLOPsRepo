import pickle
import joblib
with open('random_forest_model.pkl','rb') as f:
    m=joblib.load(f)
print(type(m))
