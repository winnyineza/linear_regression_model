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
    features: list = Field(..., min_items=4, max_items=4, description="List of 4 features for prediction")

    class Config:
        schema_extra = {
            "example": {
                "features": [0.5, 0.3, 0.2, 0.1]
            }
        }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the HTML frontend"""
    try:
        with open("index.html", "r") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading index.html: {e}")
        raise HTTPException(status_code=500, detail="Error loading frontend")

@app.get("/api/", response_class=JSONResponse)
async def api_root():
    """API root endpoint"""
    return {
        "message": "Welcome to the Global Temperature Anomaly Prediction API",
        "model_loaded": model is not None,
        "version": "1.0.0",
        "working_directory": str(os.getcwd()),
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "python_executable": sys.executable
    }

@app.post("/predict", response_class=JSONResponse)
async def predict(input_data: PredictionInput):
    """Make a prediction using the loaded model"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        features = np.array(input_data.features).reshape(1, -1)
        logger.info(f"Received input features: {input_data.features}")
        logger.info(f"Reshaped features: {features.shape}")
        
        prediction = model.predict(features)[0]
        logger.info(f"Generated prediction: {prediction}")
        
        return {
            "prediction": float(prediction),
            "input_features": input_data.features,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 