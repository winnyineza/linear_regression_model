from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
from .models.model import TemperaturePredictor

# Initialize predictor
predictor = TemperaturePredictor()

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

class PredictionInput(BaseModel):
    features: List[float]

    class Config:
        schema_extra = {
            "example": {
                "features": [0.2, 0.5, 0.7, 0.3]
            }
        }

FEATURE_NAMES = [
    "CO2 Concentration",
    "Solar Activity",
    "Ocean Temperature",
    "Atmospheric Pressure"
]

FEATURE_RANGES = {
    "CO2 Concentration": {"min": 0.0, "max": 1.0},
    "Solar Activity": {"min": 0.0, "max": 1.0},
    "Ocean Temperature": {"min": 0.0, "max": 1.0},
    "Atmospheric Pressure": {"min": 0.0, "max": 1.0}
}

@app.get("/")
def root() -> Dict:
    return {
        "message": "Welcome to the Global Temperature Anomaly Prediction API",
        "model_loaded": predictor.is_loaded,
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict",
            "model-info": "/model-info",
            "validate": "/validate"
        }
    }

@app.post("/predict")
def predict(input_data: PredictionInput):
    if len(input_data.features) != 4:
        raise HTTPException(status_code=400, detail="Exactly 4 features are required")
    
    try:
        prediction = predictor.predict(input_data.features)
        return {
            "prediction": prediction,
            "input_features": {
                FEATURE_NAMES[i]: input_data.features[i]
                for i in range(len(FEATURE_NAMES))
            },
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "feature_names": FEATURE_NAMES,
        "feature_ranges": FEATURE_RANGES,
        "model_loaded": predictor.is_loaded,
        "output_description": "Temperature Anomaly (°C)",
        "model_type": "Linear Regression"
    }

@app.post("/validate")
def validate_features(input_data: PredictionInput):
    if len(input_data.features) != 4:
        return {
            "valid": False,
            "errors": ["Exactly 4 features are required"]
        }
    
    errors = []
    for i, value in enumerate(input_data.features):
        feature_name = FEATURE_NAMES[i]
        range_info = FEATURE_RANGES[feature_name]
        if value < range_info["min"] or value > range_info["max"]:
            errors.append(
                f"{feature_name} must be between {range_info['min']} and {range_info['max']}"
            )
    
    return {
        "valid": len(errors) == 0,
        "errors": errors if errors else None,
        "validated_features": {
            FEATURE_NAMES[i]: {
                "value": input_data.features[i],
                "in_range": range_info["min"] <= input_data.features[i] <= range_info["max"]
            }
            for i, range_info in enumerate(FEATURE_RANGES.values())
        }
    } 