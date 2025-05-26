import joblib
import numpy as np
import logging
from typing import Union, Tuple
import pandas as pd

class MLFilter:
    """
    MLFilter provides a unified interface to any pre-trained ML model (XGBoost, RandomForest, etc.)
    used in the XAlgo system for:

    - Confidence scoring (triangular_rf_model.pkl)
    - Cointegration detection (cointegration_score_model.pkl)
    - Regime classification (regime_classifier.pkl)
    - Pair selection (pair_selector_model.pkl)

    Supports both classification and probabilistic prediction modes.
    """

    def __init__(self, model_path: str = "ml_model/triangular_rf_model.pkl"):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        try:
            model = joblib.load(self.model_path)
            if hasattr(model, "classes_"):
                logging.info(f"[MLFilter] ✅ Loaded model: {self.model_path} | Classes: {model.classes_}")
            else:
                logging.warning(f"[MLFilter] ⚠️ Model {self.model_path} loaded, but missing `.classes_`")
            return model
        except Exception as e:
            logging.error(f"[MLFilter] ❌ Failed to load model from {self.model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def _prepare_input(self, x_input: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Converts input to a single-row DataFrame with correct column order.
        """
        if isinstance(x_input, pd.DataFrame):
            df = x_input.copy()
        elif isinstance(x_input, np.ndarray):
            df = pd.DataFrame(x_input.reshape(1, -1))
        else:
            raise TypeError("x_input must be a DataFrame or np.ndarray")

        if df.shape[0] != 1:
            raise ValueError("x_input must be a single row")

        if hasattr(self.model, "feature_names_in_"):
            try:
                df = df.reindex(columns=self.model.feature_names_in_)
            except Exception as e:
                logging.error(f"[MLFilter] ❌ Column alignment failed: {e}")
                raise e
        else:
            logging.warning("[MLFilter] ⚠️ Model missing feature_names_in_, input order must match exactly")

        return df

    def predict_with_confidence(
        self,
        x_input: Union[np.ndarray, pd.DataFrame],
        debug: bool = False
    ) -> Tuple[float, int]:
        """
        Predicts a confidence score and directional signal from a single row of features.

        Returns:
        - confidence: float ∈ [0.0, 1.0]
        - signal: int ∈ {-1, +1}
        """
        try:
            x_df = self._prepare_input(x_input)
            probas = self.model.predict_proba(x_df)
            confidence = float(np.max(probas))
            predicted_class = int(np.argmax(probas[0]))
            signal = -1 if predicted_class == 0 else 1

            if debug:
                logging.debug(f"[MLFilter] 📊 Input: {x_df.to_dict(orient='records')[0]}")
                logging.debug(f"[MLFilter] 🔍 Class: {predicted_class}, Signal: {signal}, Confidence: {confidence:.4f}")

            return confidence, signal

        except Exception as e:
            logging.error(f"[MLFilter] ❌ Prediction error: {e}")
            return 0.0, 0

    def predict(self, x_input: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predicts class labels directly for multi-class use cases.

        Returns:
        - np.ndarray of class predictions
        """
        try:
            x_df = self._prepare_input(x_input)
            return self.model.predict(x_df)
        except Exception as e:
            logging.error(f"[MLFilter] ❌ Direct prediction error: {e}")
            return np.array([-1])
