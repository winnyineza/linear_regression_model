from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pickle
import numpy as np
from typing import List
import os
from pathlib import Path
import logging
import sys
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log Python version and environment
logger.info(f"Python version: {sys.version}")
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Current working directory: {os.getcwd()}")

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

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Looking for model at: {MODEL_PATH}")
logger.info(f"Model file exists: {MODEL_PATH.exists()}")

# Load the model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model from {MODEL_PATH}: {e}")
    model = None

class PredictionInput(BaseModel):
    features: List[float] = Field(..., min_items=4, max_items=4, description="List of 4 features for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.5, 0.3, 0.2, 0.1]
            }
        }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": str(exc)
        }
    )

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    try:
        return {
            "message": "Welcome to the Global Temperature Anomaly Prediction API",
            "model_loaded": model is not None,
            "version": "1.0.0",
            "working_directory": os.getcwd(),
            "model_path": str(MODEL_PATH),
            "model_exists": MODEL_PATH.exists(),
            "python_version": sys.version,
            "python_executable": sys.executable
        }
    except Exception as e:
        logger.error(f"Error in root endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in root endpoint: {str(e)}"
        )

@app.post("/predict")
async def predict(input_data: PredictionInput):
    """Make a prediction using the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=503,
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
    except ValueError as e:
        logger.error(f"Value error in prediction: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 