import numpy as np
import pickle

def load_model(model_path='best_model.pkl'):
    """Load the trained model"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_temperature(year, model_info=None):
    """
    Predict temperature anomaly for a given year
    
    Parameters:
    year: Year to predict for
    model_info: Loaded model information
    
    Returns:
    prediction: Predicted temperature anomaly
    """
    if model_info is None:
        model_info = load_model()
    
    # Check if it's a gradient descent model or scikit-learn model
    if isinstance(model_info, dict) and 'theta' in model_info:
        # Gradient Descent model
        theta = model_info['theta']
        X_mean = model_info['X_mean']
        X_std = model_info['X_std']
        
        # Normalize input
        year_norm = (year - X_mean) / X_std
        # Add intercept term
        X_pred = np.array([1, year_norm])
        # Make prediction
        prediction = X_pred.dot(theta)
    else:
        # Scikit-learn model
        X_pred = np.array([[year, year**2]])
        prediction = model_info.predict(X_pred)[0]
    
    return prediction

if __name__ == "__main__":
    # Example usage
    model_info = load_model()
    current_year = 2025
    prediction = predict_temperature(current_year, model_info)
    print(f"Predicted temperature anomaly for {current_year}: {prediction:.4f}°C")