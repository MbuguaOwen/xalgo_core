# utils/filters.py – MLFilter for Triangular Arbitrage Model

import joblib
import numpy as np
import logging

class MLFilter:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        try:
            model = joblib.load(self.model_path)
            logging.info(f"✅ Loaded model from {self.model_path}")
            return model
        except Exception as e:
            logging.error(f"❌ Failed to load model: {e}")
            raise

    def predict_with_confidence(self, x_input: np.ndarray):
        probas = self.model.predict_proba(x_input)
        confidence = np.max(probas)
        signal = np.argmax(probas[0]) - 1  # Map [0,1,2] → [-1,0,+1]
        return float(confidence), int(signal)
