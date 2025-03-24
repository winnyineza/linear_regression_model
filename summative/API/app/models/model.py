import pickle
import os
import numpy as np
from typing import List

class TemperaturePredictor:
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
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
            raise ValueError("Model not loaded")
        
        features_array = np.array(features).reshape(1, -1)
        return float(self.model.predict(features_array)[0])
    
    @property
    def is_loaded(self) -> bool:
        return self.model is not None 