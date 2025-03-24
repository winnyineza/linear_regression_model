# Global Temperature Anomaly Prediction

This project implements a machine learning model to predict global temperature anomalies based on various environmental factors. The project consists of a machine learning model, a FastAPI backend, and a Flutter frontend application.

## App UI Design

The application features a modern, user-friendly interface designed for ease of use and clear data visualization.

| Welcome Screen | Input Form | Form with Values | Results | History |
|:---:|:---:|:---:|:---:|:---:|
| <img src="screenshots/Home%20Screen.jpg" width="150"/> | <img src="screenshots/Screen%202%20.jpg" width="150"/> | <img src="screenshots/Screen%202%20with%20values.jpg" width="150"/> | <img src="screenshots/Result%20Screen.jpg" width="150"/> | <img src="screenshots/Prediction%20History.jpg" width="150"/> |

## Project Structure
```
linear_regression_model/
├── summative/
│   ├── linear_regression/
│   │   ├── multivariate.ipynb
│   ├── API/
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   ├── Procfile
│   ├── FlutterApp/
```

## API Endpoint
The prediction API is hosted on Render and available at:
https://temperature-anomaly-api.onrender.com

You can test the API using Swagger UI at:
https://temperature-anomaly-api.onrender.com/docs

### API Usage Example
```bash
# Get API information
curl https://temperature-anomaly-api.onrender.com/

# Make a prediction
curl -X POST https://temperature-anomaly-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 0.3, 0.2, 0.1]}'
```

## Running the Flutter App

### Prerequisites
1. Flutter SDK (>=3.0.0)
2. Dart SDK
3. A web browser (Microsoft Edge or Chrome recommended)

### Installation Steps
1. Clone this repository:
```bash
git clone [your-repo-url]
cd temperature_predictor_flutter
```

2. Install dependencies:
```bash
flutter pub get
```

3. Run the app:
```bash
# For Microsoft Edge
flutter run -d edge --web-browser-flag="--disable-web-security"

# For Chrome
flutter run -d chrome --web-browser-flag="--disable-web-security"
```

### Using the App
1. Launch the app using the commands above
2. Input values using the sliders:
   - CO2 Concentration (0-1)
   - Solar Activity (0-1)
   - Ocean Temperature (0-1)
   - Atmospheric Pressure (0-1)
3. Click "Generate Prediction" to get the temperature anomaly prediction
4. View the radar chart visualization of your inputs
5. Check the prediction history below

## Model Performance
The project compares three different models:
- Linear Regression
- Decision Trees
- Random Forest

The Linear Regression model was chosen as the best performing model based on:
- Lower Mean Squared Error (MSE)
- Better generalization on test data
- Simpler model architecture suitable for the linear nature of temperature anomaly predictions

## Video Demo
[Demo Video](https://docs.google.com/document/d/13s9qF54S_UDDHkaU5DtYedcdbeR6waaSgwVnuCjj-v0/edit?usp=sharing)

## Features

- Real-time temperature anomaly predictions
- Input validation for environmental factors
- Beautiful data visualization
- Prediction history tracking
- Modern, responsive design

## Technical Details

### Frontend (Flutter)
- Material Design 3.0
- Interactive data input forms
- Real-time validation
- Graph visualization using FL Chart
- Cross-platform compatibility

### Backend (FastAPI)
- RESTful API endpoints
- Machine learning model integration
- Data validation using Pydantic
- CORS support for web clients
- Swagger UI documentation

## API Documentation

The API is hosted at: https://temperature-anomaly-api.onrender.com/docs

Available endpoints:
- `/predict` - Make temperature predictions
- `/` - API information
- `/docs` - Swagger documentation

## Installation and Setup

```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn main:app --reload

# Run the Flutter app
cd temperature_predictor_flutter
flutter run
```

## Model Performance

The model has been trained on environmental data and achieves:
- High accuracy in temperature anomaly predictions
- Low mean squared error
- Good generalization to new data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.