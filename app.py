from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Global Temperature Anomaly Prediction API",
    description="API for predicting global temperature anomalies using machine learning models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the model file
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "summative" / "linear_regression" / "best_model.pkl"

logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Looking for model at: {MODEL_PATH}")

# Load the model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model from {MODEL_PATH}: {e}")
    model = None

class PredictionInput(BaseModel):
    features: List[float]
    
    class Config:
        schema_extra = {
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
        "working_directory": os.getcwd(),
        "model_path": str(MODEL_PATH)
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
        features = np.array(input_data.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        return {
            "prediction": float(prediction),
            "input_features": input_data.features,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction error: {str(e)}"
        ) 