import numpy as np
import pandas as pd


class FibrosisAgent:

    def __init__(self, model):

        self.name = "FibrosisAgent"
        self.model_name = "XGBoost"
        self.model = model

        # Exact features used during training
        self.features = [
            "age",
            "male",
            "weight",
            "height",
            "bmi",
            "futime"
        ]

    def predict(self, patient_data):

        # ==================================================
        # CREATE DATAFRAME WITH FEATURE NAMES
        # ==================================================

        if isinstance(patient_data, dict):

            X = pd.DataFrame(
                [patient_data],
                columns=self.features
            )

        elif isinstance(patient_data, pd.DataFrame):

            X = patient_data[
                self.features
            ].copy()

        else:

            X = pd.DataFrame(
                [patient_data],
                columns=self.features
            )

        # ==================================================
        # ENSURE CORRECT COLUMN ORDER
        # ==================================================

        X = X[self.features].copy()

        # ==================================================
        # PREDICTION
        # ==================================================

        prediction = self.model.predict(X)[0]

        # ==================================================
        # PROBABILITY
        # ==================================================

        probability = None

        if hasattr(self.model, "predict_proba"):

            probabilities = self.model.predict_proba(X)[0]

            probability = float(
                np.max(probabilities)
            )

        # ==================================================
        # RESULT
        # ==================================================

        return {
            "agent": self.name,
            "model": self.model_name,
            "prediction": str(prediction),
            "probability": probability,
            "status": "completed"
        }
