from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import pickle
import os

# Initialize FastAPI app
app = FastAPI(
    title="Global Temperature Anomaly Prediction API",
    description="API for predicting global temperature anomalies using machine learning",
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

# Load model - look for model file in the same directory as main.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")
print(f"Looking for model at: {MODEL_PATH}")
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir(os.path.dirname(__file__))}")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None

class PredictionInput(BaseModel):
    features: List[float] = Field(
        ...,
        description="List of 4 features: [CO2 Concentration, Solar Activity, Ocean Temperature, Atmospheric Pressure]",
        min_items=4,
        max_items=4,
        example=[0.5, 0.3, 0.2, 0.1]
    )

    @property
    def feature_names(self):
        return ["CO2 Concentration", "Solar Activity", "Ocean Temperature", "Atmospheric Pressure"]

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
        "model_path": MODEL_PATH,
        "current_directory": os.getcwd(),
        "directory_contents": os.listdir(os.path.dirname(__file__)),
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
            detail=f"Model not loaded. Please check server logs. Model path: {MODEL_PATH}"
        )
    
    try:
        # Validate input ranges
        for i, value in enumerate(input_data.features):
            if not 0 <= value <= 1:
                raise ValueError(
                    f"{input_data.feature_names[i]} must be between 0 and 1, got {value}"
                )
        
        # Convert input features to numpy array and reshape for prediction
        features = np.array(input_data.features).reshape(1, -1)
        
        # Make prediction
        prediction = float(model.predict(features)[0])
        
        return {
            "prediction": prediction,
            "input_features": input_data.features,
            "feature_names": input_data.feature_names,
            "status": "success"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000))) 