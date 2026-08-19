# src/serve.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="MLOps Predictive Inference Service")

MODEL_PATH = "models/production_model.joblib"
BUFFER_PATH = "data/buffer.csv"

class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

def get_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model belum tersedia. Jalankan training terlebih dahulu.")
    return joblib.load(MODEL_PATH)

def log_request(data: dict):
    df_new = pd.DataFrame([data])
    if not os.path.exists(BUFFER_PATH):
        df_new.to_csv(BUFFER_PATH, index=False)
    else:
        df_new.to_csv(BUFFER_PATH, mode="a", header=False, index=False)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": os.path.exists(MODEL_PATH)}

@app.post("/predict")
def predict(features: HousingFeatures):
    model = get_model()
    data_dict = features.dict()
    df_features = pd.DataFrame([data_dict])
    
    prediction = model.predict(df_features)[0]
    
    # Simpan fitur ke buffer untuk drift monitoring
    log_request(data_dict)
    
    return {
        "prediction_MedHouseVal": float(prediction),
        "status": "success"
    }