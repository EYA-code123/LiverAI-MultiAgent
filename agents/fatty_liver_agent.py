"""
Fatty Liver Agent
Uses the trained LightGBM model for fatty liver classification.
"""

import numpy as np


class FattyLiverAgent:

    def __init__(self, model):
        self.name = "FattyLiverAgent"
        self.model_name = "LightGBM"
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
