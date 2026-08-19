import numpy as np
import pandas as pd


class FibrosisAgent:

    def __init__(self, model):

        self.name = "FibrosisAgent"
        self.model_name = "XGBoost"
        self.model = model

        self.features = [
            "age",
            "male",
            "weight",
            "height",
            "bmi",
            "futime"
        ]

    def predict(self, patient_data):

        if isinstance(patient_data, dict):

            X = pd.DataFrame([patient_data])

        elif isinstance(patient_data, pd.DataFrame):

            X = patient_data.copy()

        else:

            X = pd.DataFrame(
                [patient_data],
                columns=self.features
            )

        missing_features = [
            feature
            for feature in self.features
            if feature not in X.columns
        ]

        if missing_features:

            raise ValueError(
                f"Missing features: {missing_features}"
            )

        X = X[self.features].copy()

        prediction = self.model.predict(X)[0]

        probability = None

        if hasattr(self.model, "predict_proba"):

            probabilities = self.model.predict_proba(X)[0]

            probability = float(
                np.max(probabilities)
            )

        return {
            "agent": self.name,
            "model": self.model_name,
            "prediction": str(prediction),
            "probability": probability,
            "status": "completed"
        }
