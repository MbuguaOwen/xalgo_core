# filters/ml_filter.py – MLFilter for Triangular Arbitrage (Production-Aligned)

import joblib
import numpy as np
import logging

class MLFilter:
    """
    Uses a calibrated RandomForestClassifier to filter trade signals based on real-time features.
    Maps model output [0,1,2] to [-1, 0, 1] (Sell, Hold, Buy).
    """

    def __init__(self, model_path: str = "models/triangular_rf_model.pkl"):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        try:
            model = joblib.load(self.model_path)
            logging.info(f"[MLFilter] ✅ Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            logging.error(f"[MLFilter] ❌ Error loading model: {e}")
            raise

    def predict_with_confidence(self, x_input: np.ndarray):
        """
        Predicts trade signal and confidence using the model.

        Parameters:
            x_input (np.ndarray): 2D array of features (shape: [1, n_features])

        Returns:
            Tuple[float, int]: confidence score (0–1), trade signal (-1, 0, 1)
        """
        probas = self.model.predict_proba(x_input)
        confidence = np.max(probas)
        signal = np.argmax(probas[0]) - 1  # Convert class 0,1,2 to -1,0,+1
        return float(confidence), int(signal)
