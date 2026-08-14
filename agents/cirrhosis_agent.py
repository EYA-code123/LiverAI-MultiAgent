"""
Cirrhosis Agent
Uses the trained XGBoost model for cirrhosis classification.
"""

import numpy as np


class CirrhosisAgent:

    def __init__(self, model):
        """
        Parameters
        ----------
        model : trained XGBoost model
        """
        self.name = "CirrhosisAgent"
        self.model_name = "XGBoost"
        self.model = model

    def predict(self, patient_data):
        """
        Run cirrhosis prediction.

        patient_data must contain the features expected
        by the trained XGBoost model.
        """

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
