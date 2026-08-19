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

        # ==================================================
        # CREATE DATAFRAME
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

        # Ensure correct order
        X = X[self.features].copy()

        # Convert values to numeric
        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

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
                max(probabilities)
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
