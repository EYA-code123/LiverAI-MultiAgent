%%writefile /content/LiverAI-MultiAgent/agents/fatty_liver_agent.py

import numpy as np
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

    # ==========================================================
    # 1. Convertir en DataFrame
    # ==========================================================

    if isinstance(patient_data, dict):
        X = pd.DataFrame([patient_data])

    elif isinstance(patient_data, pd.DataFrame):
        X = patient_data.copy()

    else:
        X = pd.DataFrame(
            [patient_data],
            columns=self.features
        )

    # ==========================================================
    # 2. Vérifier les features
    # ==========================================================

    for feature in self.features:
        if feature not in X.columns:
            X[feature] = np.nan

    X = X[self.features].copy()

    # ==========================================================
    # 3. Conversion numérique
    # ==========================================================

    for feature in self.features:
        X[feature] = pd.to_numeric(
            X[feature],
            errors="coerce"
        )

    # ==========================================================
    # 4. Prediction
    # ==========================================================

    prediction = self.model.predict(X)[0]

    # ==========================================================
    # 5. Probability
    # ==========================================================

    probability = None

    if hasattr(self.model, "predict_proba"):

        probabilities = self.model.predict_proba(X)[0]

        probability = float(np.max(probabilities))

    # ==========================================================
    # 6. Résultat
    # ==========================================================

    return {
        "agent": self.name,
        "model": self.model_name,
        "prediction": str(prediction),
        "probability": probability,
        "status": "completed"
    }
