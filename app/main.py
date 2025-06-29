from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
import os

app = FastAPI()

#загрузка артефактов
def load_artifacts():
    artifacts_path = os.path.dirname(__file__)
    return {
        'model': joblib.load(f"{artifacts_path}/artifacts/model.pkl"),
        'encoder': joblib.load(f"{artifacts_path}/artifacts/encoder.pkl"),
        'scaler': joblib.load(f"{artifacts_path}/artifacts/scaler.pkl"),
        'features': joblib.load(f"{artifacts_path}/artifacts/features.pkl")
    }

artifacts = load_artifacts()


class PatientData(BaseModel):
    age: float
    cholesterol: float
    heart_rate: float
    diabetes: int
    smoking: int
    obesity: int
    alcohol_consumption: int
    exercise_hours_per_week: float
    previous_heart_problems: int
    stress_level: int
    sedentary_hours_per_day: float
    bmi: float
    triglycerides: float
    sleep_hours_per_day: float
    blood_sugar: float
    ck_mb: float
    troponin: float
    systolic_blood_pressure: float
    diastolic_blood_pressure: float

@app.post("/predict")
def predict(data: PatientData):
    try:
        input_df = pd.DataFrame([data.dict()])
        
        X_cat = artifacts['encoder'].transform(input_df[artifacts['features']['categorical']])
        X_num = artifacts['scaler'].transform(input_df[artifacts['features']['numerical']])
        X = np.hstack([X_num, X_cat])
        
        prediction = artifacts['model'].predict(X)[0]
        probability = artifacts['model'].predict_proba(X)[0][1]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "risk": "high" if prediction == 1 else "low"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)