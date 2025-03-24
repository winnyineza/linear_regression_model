import pickle
import os
import numpy as np
from typing import List

class TemperaturePredictor:
    def __init__(self):
        self.model = None
        # Get the absolute path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, "best_model.pkl")
        print(f"Current directory: {current_dir}")
        print(f"Looking for model at: {self.model_path}")
        print(f"Directory contents: {os.listdir(current_dir)}")
        self.load_model()
    
    def load_model(self):
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            print(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            print(f"Error loading model from {self.model_path}: {e}")
            self.model = None
    
    def predict(self, features: List[float]) -> float:
        if self.model is None:
            raise ValueError(f"Model not loaded. Tried path: {self.model_path}")
        
        features_array = np.array(features).reshape(1, -1)
        return float(self.model.predict(features_array)[0])
    
    @property
    def is_loaded(self) -> bool:
        return self.model is not None 