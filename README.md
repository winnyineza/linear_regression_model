# Global Temperature Anomaly Prediction API

This FastAPI application provides endpoints for predicting global temperature anomalies using a machine learning model.

## Setup and Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the API Locally

Start the API server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- Interactive API docs (Swagger UI): `http://localhost:8000/docs`
- Alternative API docs (ReDoc): `http://localhost:8000/redoc`

## API Endpoints

### 1. Root Endpoint
```http
GET /
```

Returns API information and status.

Example response:
```json
{
    "message": "Welcome to the Global Temperature Anomaly Prediction API",
    "model_loaded": true,
    "version": "1.0.0",
    "working_directory": "/path/to/working/directory",
    "model_path": "/path/to/model.pkl",
    "model_exists": true,
    "python_version": "3.x.x",
    "python_executable": "/path/to/python"
}
```

### 2. Prediction Endpoint
```http
POST /predict
```

Make a prediction using the model.

Request body:
```json
{
    "features": [0.5, 0.3, 0.2, 0.1]
}
```

Example response:
```json
{
    "prediction": 0.42,
    "input_features": [0.5, 0.3, 0.2, 0.1],
    "status": "success"
}
```

## Testing the API

### Using the Test Script

Run the provided test script to verify all endpoints:
```bash
python test_api.py
```

### Using cURL

1. Test the root endpoint:
```bash
curl http://localhost:8000/
```

2. Make a prediction:
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [0.5, 0.3, 0.2, 0.1]}'
```

### Using Python requests

```python
import requests

# Test root endpoint
response = requests.get("http://localhost:8000/")
print(response.json())

# Make a prediction
data = {"features": [0.5, 0.3, 0.2, 0.1]}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## Error Handling

The API includes comprehensive error handling for various scenarios:

1. Invalid input (400 Bad Request)
2. Model not loaded (503 Service Unavailable)
3. Server errors (500 Internal Server Error)

Example error response:
```json
{
    "detail": "Invalid input: Input must contain exactly 4 features"
}
```

## Deployment

The API is configured for deployment on Render. The following files are used for deployment:

- `render.yaml`: Deployment configuration
- `Procfile`: Process configuration for the web service
- `requirements.txt`: Python dependencies

## Logging

The API includes comprehensive logging for debugging and monitoring:
- Application startup information
- Model loading status
- Request processing
- Error details and stack traces 