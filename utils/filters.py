# utils/filters.py – MLFilter for Triangular Arbitrage (Binary Directional Model)

import joblib
import numpy as np
import logging

class MLFilter:
    """
    Uses a calibrated binary classifier (e.g., XGBoost or RandomForest)
    to predict directional trade signals from real-time input features.

    Model is trained on: -1 (short) → class 0, +1 (long) → class 1
    Returns:
        - confidence ∈ [0.0, 1.0]
        - signal ∈ {–1, +1}
    """

    def __init__(self, model_path: str = "ml_model/triangular_rf_model.pkl"):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        try:
            model = joblib.load(self.model_path)
            if hasattr(model, "classes_"):
                logging.info(f"[MLFilter] ✅ Model loaded from {self.model_path} | Classes: {model.classes_}")
            else:
                logging.warning("[MLFilter] ⚠️ Model loaded but missing .classes_")
            return model
        except Exception as e:
            logging.error(f"[MLFilter] ❌ Failed to load model: {e}")
            raise

    def predict_with_confidence(self, x_input: np.ndarray):
        """
        Predicts trade signal and confidence.

        Parameters:
            x_input (np.ndarray or pd.DataFrame): shape (1, n_features)

        Returns:
            Tuple[float, int]: confidence, signal (∈ {–1, +1})
        """
        try:
            probas = self.model.predict_proba(x_input)
            confidence = float(np.max(probas))
            predicted_class = int(np.argmax(probas[0]))

            # 🔁 Remap model output: 0 → -1, 1 → +1
            signal = -1 if predicted_class == 0 else 1
            return confidence, signal

        except Exception as e:
            logging.error(f"[MLFilter] ❌ Prediction error: {e}")
            return 0.0, 0  # fallback: no trade
