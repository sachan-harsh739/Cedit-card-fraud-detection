from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import os
import time

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Set up logging for audit trail
logger = logging.getLogger("fraud_api")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('logs/api_audit.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

app = FastAPI(title="Ultimate Enterprise Fraud API",
              description="Real-time Credit Card Fraud Detection with Audit Tracking.")

model, scaler, expected_features, opt_threshold, version = None, None, None, 0.5, "UNKNOWN"

try:
    model_data = joblib.load("models/fraud_model.pkl")
    model = model_data['model']
    scaler = model_data['scaler']
    expected_features = model_data['features']
    opt_threshold = model_data.get('optimal_threshold', 0.5)
    version = model_data.get('version', '1.0')
except Exception as e:
    logger.error(f"Startup error loading model: {e}")

class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float

@app.on_event("startup")
def startup_event():
    if model is None:
        logger.warning("API started without a valid model found in models/fraud_model.pkl.")
    else:
        logger.info(f"API startup successful. Loaded model version {version} with optimal threshold {opt_threshold:.4f}.")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/health")
def health_check():
    status = "healthy" if model is not None else "degraded"
    return {"status": status, "model_version": version}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if model is None:
        logger.error("Predict endpoint called but model is unavailable.")
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
        
    try:
        data_dict = transaction.dict()
        df = pd.DataFrame([data_dict])
        
        # Feature Engineering equivalent
        df['Hour_of_Day'] = (df['Time'] // 3600) % 24
        cols_to_scale = ['Amount', 'Time', 'Hour_of_Day']
        scaled_features = scaler.transform(df[cols_to_scale])
        
        df['Scaled_Amount'] = scaled_features[:, 0]
        df['Scaled_Time'] = scaled_features[:, 1]
        df['Scaled_Hour'] = scaled_features[:, 2]
        df.drop(cols_to_scale, axis=1, inplace=True)
        
        df = df[expected_features]
        
        prob = None
        if hasattr(model, 'predict_proba'):
            prob = float(model.predict_proba(df)[0][1])
            pred = 1 if prob >= opt_threshold else 0
        else:
            pred = int(model.predict(df)[0])
            prob = float(pred)
            
        verdict = "Fraud" if pred == 1 else "Legit"
        
        # Audit Log
        logger.info(f"Transaction scored | Prob: {prob:.4f} | Prediction: {verdict} | Inputs: {data_dict}")
        
        return {
            "prediction": verdict,
            "fraud_probability": prob,
            "optimal_threshold_applied": opt_threshold,
            "model_version": version
        }
        
    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
