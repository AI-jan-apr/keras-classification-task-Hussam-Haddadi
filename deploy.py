from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load('model_weights.pkl')
scaler = joblib.load('scaler_weights.pkl')

app = FastAPI(title="Cancer Prediction API")

class PredictionInput(BaseModel):
    features: list  

@app.post("/predict")
def predict_cancer(data: PredictionInput):
    input_array = np.array(data.features).reshape(1, -1)
    
    scaled_data = scaler.transform(input_array)
    
    prediction = (model.predict(scaled_data) > 0.5).astype("int32")
    
    class_label = "Malignant (1)" if prediction[0][0] == 1 else "Benign (0)"
    
    return {
        "prediction_code": int(prediction[0][0]),
        "result": class_label
    }

@app.get("/")
def home():
    return {"message": "Cancer Prediction API is running. Go to /docs for testing."}