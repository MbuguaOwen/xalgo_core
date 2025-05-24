# adaptive_filters.py – EWMA and Kalman Filter Toolkit for Spread Normalization

import numpy as np

class EWMA:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

class KalmanFilter:
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2):
        self.x = 0.0
        self.P = 1.0
        self.Q = process_variance
        self.R = measurement_variance

    def update(self, z):
        # Predict
        self.P = self.P + self.Q
        # Update
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x
