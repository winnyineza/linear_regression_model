from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pickle
import logging
from pathlib import Path
import sys
import os
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Global Temperature Anomaly Prediction API",
    description="API for predicting global temperature anomalies using machine learning",
    version="1.0.0"
)

# Add CORS middleware with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"]
)

# Get base directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "summative" / "linear_regression" / "best_model.pkl"

# Load model at startup
model = None
try:
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Looking for model at: {MODEL_PATH}")
    logger.info(f"Model file exists: {MODEL_PATH.exists()}")
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Model attributes: {dir(model)}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.error(f"Current working directory: {os.getcwd()}")
    logger.error(f"Model path: {MODEL_PATH}")
    logger.error(f"Model path exists: {MODEL_PATH.exists()}")
    logger.error(f"Model path is absolute: {MODEL_PATH.is_absolute()}")

class PredictionInput(BaseModel):
    features: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5, 0.3, 0.2, 0.1]
            }
        }

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "Welcome to the Global Temperature Anomaly Prediction API",
        "model_loaded": model is not None,
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict"
        }
    }

@app.post("/predict")
async def predict(input_data: PredictionInput):
    """Make a prediction using the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert input features to numpy array and reshape for prediction
        features = np.array(input_data.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return {
            "prediction": float(prediction),
            "input_features": input_data.features,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 