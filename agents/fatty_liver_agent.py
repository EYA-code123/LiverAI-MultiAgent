```python
# ==========================================================
# Fatty Liver Agent
# ==========================================================

import pandas as pd
import numpy as np


class FattyLiverAgent:

    def __init__(self, model):

        self.name = "FattyLiverAgent"
        self.model_name = "LightGBM"
        self.model = model

        # Features EXACTEMENT utilisées pendant l'entraînement
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
        # Create DataFrame with feature names
        # --------------------------------------------------

        if isinstance(patient_data, dict):

            X = pd.DataFrame(
                [patient_data],
                columns=self.features
            )

        elif isinstance(patient_data, pd.DataFrame):

            X = patient_data[self.features].copy()

        else:

            X = pd.DataFrame(
                [patient_data],
                columns=self.features
            )

        # --------------------------------------------------
        # Ensure correct column order
        # --------------------------------------------------

        X = X[self.features].copy()

        # --------------------------------------------------
        # Convert values to numeric
        # --------------------------------------------------

        for col in self.features:

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
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
```
