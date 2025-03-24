import pickle
import numpy as np
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    """Test model loading and prediction"""
    try:
        # Get model path
        model_path = Path("summative/linear_regression/best_model.pkl")
        logger.info(f"Testing model at: {model_path}")
        
        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model attributes: {dir(model)}")
        
        # Test prediction
        test_features = np.array([[0.5, 0.3, 0.2, 0.1]])
        prediction = model.predict(test_features)[0]
        logger.info(f"Test prediction: {prediction}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return False

if __name__ == "__main__":
    test_model() 