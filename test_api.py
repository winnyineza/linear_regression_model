import requests
import json
import logging
import time
from requests.exceptions import RequestException

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoint
BASE_URL = "http://localhost:8000"

def wait_for_server(max_retries=5, delay=2):
    """Wait for the server to be ready"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/")
            if response.status_code == 200:
                logger.info("Server is ready!")
                return True
        except RequestException:
            if i < max_retries - 1:
                logger.info(f"Server not ready, retrying in {delay} seconds... (attempt {i+1}/{max_retries})")
                time.sleep(delay)
            else:
                logger.error("Server is not responding after maximum retries")
                return False
    return False

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
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        return False

def test_predict_endpoint():
    """Test the prediction endpoint"""
    # Test data
    test_data = {
        "features": [0.5, 0.3, 0.2, 0.1]
    }
    
    try:
        logger.info(f"Sending prediction request with data: {json.dumps(test_data)}")
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        
        if response.status_code != 200:
            logger.error(f"Response content: {response.text}")
            return False
            
        data = response.json()
        logger.info("Prediction endpoint test successful")
        logger.info(f"Response: {json.dumps(data, indent=2)}")
        return True
    except Exception as e:
        logger.error(f"Prediction endpoint test failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        return False

def test_invalid_input():
    """Test the prediction endpoint with invalid input"""
    # Invalid test data (wrong number of features)
    test_data = {
        "features": [0.5, 0.3]  # Only 2 features instead of 4
    }
    
    try:
        logger.info(f"Sending invalid input test with data: {json.dumps(test_data)}")
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response content: {response.text}")
        
        # Accept both 400 (manual validation) and 422 (FastAPI validation) as valid error responses
        if response.status_code in [400, 422]:
            logger.info("Invalid input test successful (got expected error)")
            return True
        else:
            logger.error(f"Invalid input test failed (expected 400 or 422, got {response.status_code})")
            return False
    except Exception as e:
        logger.error(f"Invalid input test failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        return False

def run_all_tests():
    """Run all tests"""
    logger.info("Starting API tests...")
    
    # Wait for server to be ready
    if not wait_for_server():
        logger.error("Cannot proceed with tests - server is not responding")
        return
    
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