# Global Temperature Anomaly Prediction

This project implements a machine learning model to predict global temperature anomalies based on various environmental factors. The project consists of a machine learning model, a FastAPI backend, and a Flutter frontend application.
## App UI Design
image.png



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