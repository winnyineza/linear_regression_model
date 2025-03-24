# Global Temperature Anomaly Predictor

This project predicts global temperature anomalies using machine learning models. It includes a trained model, API endpoint, and Flutter mobile application.

## Project Structure

```
linear_regression_model/
├── summative/
│   ├── linear_regression/
│   │   ├── multivariate.ipynb
│   ├── API/
│   │   ├── main.py
│   │   ├── requirements.txt
│   ├── FlutterApp/
```

## API Endpoint

Public API URL: [YOUR_RENDER_URL_HERE]

The API accepts POST requests at `/predict` with the following format:
```json
{
    "year": 2025
}
```

## Running the Mobile App

1. Install Flutter SDK (https://flutter.dev/docs/get-started/install)
2. Clone this repository
3. Navigate to the FlutterApp directory
4. Run `flutter pub get` to install dependencies
5. Update the API endpoint in `lib/screens/prediction_screen.dart`
6. Run `flutter run` to start the app

## Video Demo

Watch the project demonstration: [YOUR_YOUTUBE_LINK_HERE]

## Model Performance

The project compares three models:
- Linear Regression
- Decision Trees
- Random Forest

[Add your model performance comparison results here] 