from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
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
        "model_loaded": predictor.is_loaded,
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict"
        }
    }

@app.post("/predict")
async def predict(input_data: PredictionInput):
    """Make a prediction using the loaded model"""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Validate input ranges
        for i, value in enumerate(input_data.features):
            if not 0 <= value <= 1:
                raise ValueError(
                    f"{input_data.feature_names[i]} must be between 0 and 1, got {value}"
                )
        
        # Make prediction
        prediction = predictor.predict(input_data.features)
        
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