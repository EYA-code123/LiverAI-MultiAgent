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

    # Features exactes utilisées pendant l'entraînement
    X = pd.DataFrame(
        [patient_data],
        columns=self.features
    )

    # S'assurer que toutes les colonnes sont numériques
    X = X.apply(pd.to_numeric, errors="coerce")

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
