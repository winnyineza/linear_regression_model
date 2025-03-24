from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List
import os

app = FastAPI(
    title="Global Temperature Anomaly Prediction API",
    description="API for predicting global temperature anomalies using machine learning models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the model
try:
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'linear_regression', 'best_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class PredictionInput(BaseModel):
    features: List[float]
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.5, 0.3, 0.2, 0.1]  # Example input features
            }
        }

@app.get("/")
async def root():
    return {"message": "Welcome to the Global Temperature Anomaly Prediction API"}

@app.post("/predict")
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input features to numpy array and reshape for prediction
        features = np.array(input_data.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return {
            "prediction": float(prediction),
            "input_features": input_data.features
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)