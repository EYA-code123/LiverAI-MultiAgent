%%writefile /content/LiverAI-MultiAgent/agents/fatty_liver_agent.py

import pandas as pd
import numpy as np


class FattyLiverAgent:

    def __init__(self, model):

        self.model = model
        self.name = "FattyLiverAgent"
        self.model_name = "LightGBM"

        # Exact features used during training
        self.features = [
            "mcv",
            "alkphos",
            "sgpt",
            "sgot",
            "gammagt",
            "drinks"
        ]

    def predict(self, patient_data):

        # --------------------------------------------------
        # Create DataFrame WITH feature names
        # --------------------------------------------------

        if isinstance(patient_data, dict):

            X = pd.DataFrame(
                [[
                    patient_data.get("mcv"),
                    patient_data.get("alkphos"),
                    patient_data.get("sgpt"),
                    patient_data.get("sgot"),
                    patient_data.get("gammagt"),
                    patient_data.get("drinks")
                ]],
                columns=self.features
            )

        else:

            X = pd.DataFrame(
                [patient_data],
                columns=self.features
            )

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        prediction = self.model.predict(X)[0]

        # --------------------------------------------------
        # Probability
        # --------------------------------------------------

        probability = None

        if hasattr(self.model, "predict_proba"):

            probabilities = self.model.predict_proba(X)[0]

            probability = float(
                np.max(probabilities)
            )

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        return {

            "agent": self.name,

            "model": self.model_name,

            "prediction": str(prediction),

            "probability": probability,

            "status": "completed"
        }
