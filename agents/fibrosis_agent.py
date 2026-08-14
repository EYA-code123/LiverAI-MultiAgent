"""
Fibrosis Prediction Agent
Uses the trained XGBoost model for fibrosis prediction.
"""

import numpy as np


class FibrosisAgent:

    def __init__(self, model):
        self.name = "FibrosisAgent"
        self.model_name = "XGBoost"
        self.model = model

    def predict(self, patient_data):

        X = np.asarray(patient_data)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        prediction = self.model.predict(X)[0]

        probability = None

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)[0]
            probability = float(np.max(probabilities))

        return {
            "agent": self.name,
            "model": self.model_name,
            "prediction": str(prediction),
            "probability": probability,
            "status": "completed"
        }
