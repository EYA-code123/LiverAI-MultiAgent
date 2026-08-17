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

    import pandas as pd
    import numpy as np

    # ======================================================
    # PREPARE INPUT AS DATAFRAME
    # ======================================================

    if isinstance(patient_data, dict):

        X = pd.DataFrame([patient_data])

    elif isinstance(patient_data, pd.DataFrame):

        X = patient_data.copy()

    else:

        X = pd.DataFrame(
            [patient_data],
            columns=self.features
        )

    # ======================================================
    # ENSURE CORRECT FEATURE ORDER
    # ======================================================

    X = X[self.features].copy()

    # ======================================================
    # CONVERT NUMERIC FEATURES
    # ======================================================

    for col in self.features:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    # ======================================================
    # PREDICTION
    # ======================================================

    prediction = self.model.predict(X)[0]

    # ======================================================
    # PROBABILITY
    # ======================================================

    probability = None

    if hasattr(self.model, "predict_proba"):

        probabilities = self.model.predict_proba(X)[0]

        probability = float(
            np.max(probabilities)
        )

    # ======================================================
    # RESULT
    # ======================================================

    return {

        "agent": self.name,

        "model": self.model_name,

        "prediction": str(prediction),

        "probability": probability,

        "status": "completed"
    }
