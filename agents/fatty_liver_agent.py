"""
Fatty Liver Agent
BUPA Liver Disorders + LightGBM
"""

import pandas as pd


class FattyLiverAgent:

    def __init__(self, model):

        self.name = "FattyLiverAgent"
        self.model_name = "LightGBM"
        self.model = model

        self.features = [
            "mcv",
            "alkphos",
            "sgpt",
            "sgot",
            "gammagt",
            "drinks"
        ]

    def predict(self, patient_data):

        # Create DataFrame with the exact training feature names
        if isinstance(patient_data, dict):

            X = pd.DataFrame(
                [patient_data],
                columns=self.features
            )

        else:

            X = pd.DataFrame(
                [patient_data],
                columns=self.features
            )

        # Prediction
        prediction = self.model.predict(X)[0]

        # Probability
        probability = None

        if hasattr(self.model, "predict_proba"):

            probabilities = self.model.predict_proba(X)[0]

            probability = float(max(probabilities))

        return {
            "agent": self.name,
            "model": self.model_name,
            "prediction": str(prediction),
            "probability": probability,
            "status": "completed"
        }
