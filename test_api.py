import requests
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoint
BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    """Test the root endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        data = response.json()
        logger.info("Root endpoint test successful")
        logger.info(f"Response: {json.dumps(data, indent=2)}")
        return True
    except Exception as e:
        logger.error(f"Root endpoint test failed: {e}")
        return False

def test_predict_endpoint():
    """Test the prediction endpoint"""
    # Test data
    test_data = {
        "features": [0.5, 0.3, 0.2, 0.1]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        logger.info("Prediction endpoint test successful")
        logger.info(f"Response: {json.dumps(data, indent=2)}")
        return True
    except Exception as e:
        logger.error(f"Prediction endpoint test failed: {e}")
        return False

def test_invalid_input():
    """Test the prediction endpoint with invalid input"""
    # Invalid test data (wrong number of features)
    test_data = {
        "features": [0.5, 0.3]  # Only 2 features instead of 4
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 400:
            logger.info("Invalid input test successful (got expected error)")
            return True
        else:
            logger.error("Invalid input test failed (didn't get expected error)")
            return False
    except Exception as e:
        logger.error(f"Invalid input test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    logger.info("Starting API tests...")
    
    tests = [
        ("Root endpoint", test_root_endpoint),
        ("Prediction endpoint", test_predict_endpoint),
        ("Invalid input", test_invalid_input)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        if not test_func():
            all_passed = False
            logger.error(f"{test_name} test failed!")
    
    if all_passed:
        logger.info("\nAll tests passed successfully!")
    else:
        logger.error("\nSome tests failed!")

if __name__ == "__main__":
    run_all_tests() 