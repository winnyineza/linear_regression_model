from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys
import os
from pathlib import Path

# Add parent directory to path to import prediction function
sys.path.append(str(Path(__file__).parent.parent))
from summative.API.prediction import predict_temperature, load_model

app = FastAPI(
    title="Global Temperature Anomaly Prediction API",
    description="API for predicting global temperature anomalies based on year",
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

# Load model at startup
model = load_model('../best_model.pkl')

class PredictionInput(BaseModel):
    year: int = Field(
        ...,
        description="Year for prediction (e.g., 2025)",
        ge=1880,  # Minimum year (adjust based on your training data)
        le=2100   # Maximum year (adjust based on your requirements)
    )

class PredictionOutput(BaseModel):
    year: int
    predicted_anomaly: float
    units: str = "°C"

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        prediction = predict_temperature(input_data.year, model)
        return PredictionOutput(
            year=input_data.year,
            predicted_anomaly=float(prediction)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 