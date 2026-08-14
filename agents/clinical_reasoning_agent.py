"""
Clinical Reasoning Agent
Uses the trained TabNet model for clinical liver disorder reasoning.
"""

import numpy as np


class ClinicalReasoningAgent:

    def __init__(self, model):
        self.name = "ClinicalReasoningAgent"
        self.model_name = "TabNet"
        self.model = model

    def predict(self, patient_data):

        X = np.asarray(patient_data)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        prediction = self.model.predict(X)[0]

        probability = None

        # TabNet models may expose predict_proba()
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)

            if probabilities is not None:
                probability = float(np.max(probabilities[0]))

        return {
            "agent": self.name,
            "model": self.model_name,
            "prediction": str(prediction),
            "probability": probability,
            "status": "completed"
        }
