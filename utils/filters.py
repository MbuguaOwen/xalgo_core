# utils/filters.py – MLFilter for Triangular Arbitrage (Binary Directional Model)

import joblib
import numpy as np
import logging
from typing import Union, Tuple
import pandas as pd


class MLFilter:
    """
    MLFilter wraps a calibrated binary classifier (e.g., XGBoost, RandomForest)
    trained to detect profitable spread reversion trades in a triangular arbitrage setup.

    Expected model behavior:
        - Class 0 → signal -1 (SHORT)
        - Class 1 → signal +1 (LONG)

    Outputs:
        - Confidence ∈ [0.0, 1.0]
        - Signal ∈ {–1, +1}
    """

    def __init__(self, model_path: str = "ml_model/triangular_rf_model.pkl"):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        try:
            model = joblib.load(self.model_path)
            if hasattr(model, "classes_"):
                logging.info(
                    f"[MLFilter] ✅ Model loaded: {self.model_path} | Classes: {model.classes_}"
                )
            else:
                logging.warning("[MLFilter] ⚠️ Model loaded, but missing `.classes_` attribute")
            return model
        except Exception as e:
            logging.error(f"[MLFilter] ❌ Failed to load model from {self.model_path}: {e}")
            raise RuntimeError(f"Failed to load ML model: {e}")

    def predict_with_confidence(
        self,
        x_input: Union[np.ndarray, pd.DataFrame],
        debug: bool = False
    ) -> Tuple[float, int]:
        """
        Predicts a trading signal and confidence score from input features.

        Parameters:
            x_input (pd.DataFrame or np.ndarray): shape (1, n_features)
            debug (bool): whether to log inputs and outputs at DEBUG level

        Returns:
            Tuple:
                confidence (float): max predicted probability ∈ [0.0, 1.0]
                signal (int): trading signal, –1 (short) or +1 (long)
        """
        try:
            if isinstance(x_input, pd.DataFrame):
                if x_input.shape[0] != 1:
                    raise ValueError("x_input must be a single row")
            elif isinstance(x_input, np.ndarray):
                x_input = pd.DataFrame(x_input.reshape(1, -1))
            else:
                raise TypeError("x_input must be a pandas DataFrame or numpy array")

            probas = self.model.predict_proba(x_input)
            confidence = float(np.max(probas))
            predicted_class = int(np.argmax(probas[0]))

            # Remap XGBoost class → trading signal
            signal = -1 if predicted_class == 0 else 1

            if debug:
                logging.debug(f"[MLFilter] 📅 Input: {x_input.to_dict(orient='records')[0]}")
                logging.debug(
                    f"[MLFilter] 🔍 Prediction: class={predicted_class}, "
                    f"signal={signal}, confidence={confidence:.4f}"
                )

            return confidence, signal

        except Exception as e:
            logging.error(f"[MLFilter] ❌ Prediction error: {e}")
            return 0.0, 0  # fallback: no trade
