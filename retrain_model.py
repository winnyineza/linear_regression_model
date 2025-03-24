import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
from pathlib import Path
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_model():
    """Create and save a sample LinearRegression model"""
    try:
        # Create a simple linear regression model
        model = LinearRegression()
        
        # Generate sample training data
        X = np.random.rand(100, 4)  # 100 samples, 4 features
        y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 100)  # Simple target with noise
        
        # Train the model
        logger.info("Training model...")
        model.fit(X, y)
        
        # Create directory if it doesn't exist
        save_dir = Path("summative/linear_regression")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        model_path = save_dir / "best_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved successfully to {model_path}")
        logger.info(f"Model path exists: {model_path.exists()}")
        logger.info(f"Model path is absolute: {model_path.is_absolute()}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Test the saved model
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        
        test_input = np.array([[0.5, 0.3, 0.2, 0.1]])
        prediction = loaded_model.predict(test_input)[0]
        logger.info(f"Test prediction with input {test_input[0]}: {prediction}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        logger.error(f"Current working directory: {os.getcwd()}")
        return False

if __name__ == "__main__":
    create_sample_model() 