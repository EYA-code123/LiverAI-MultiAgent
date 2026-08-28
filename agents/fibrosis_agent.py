import numpy as np
import pandas as pd


class FibrosisAgent:

    def __init__(self, model):

        self.name = "Fibrosis Agent"
        self.model_name = "XGBoost / Random Forest"

        self.model = model

        self.features = [
            "age",
            "male",
            "weight",
            "height",
            "bmi",
            "futime",
            "days",
            "test",
            "value"
        ]

    def predict(self, patient_data):

        try:

            if patient_data is None:
                raise ValueError("patient_data is None")

            # --------------------------------------------------
            # DATAFRAME
            # --------------------------------------------------

            if isinstance(patient_data, dict):

                X = pd.DataFrame([patient_data])

            elif isinstance(patient_data, pd.DataFrame):

                X = patient_data.copy()

            else:

                X = pd.DataFrame(
                    [patient_data],
                    columns=self.features
                )

            # --------------------------------------------------
            # ADD MISSING FEATURES
            # --------------------------------------------------

            for feature in self.features:

                if feature not in X.columns:
                    X[feature] = np.nan

            # --------------------------------------------------
            # USE MODEL FEATURES WHEN AVAILABLE
            # --------------------------------------------------

            if hasattr(self.model, "feature_names_in_"):

                expected_features = list(
                    self.model.feature_names_in_
                )

                for feature in expected_features:

                    if feature not in X.columns:
                        X[feature] = np.nan

                X = X[expected_features].copy()

            else:

                X = X[self.features].copy()

            # --------------------------------------------------
            # PREDICTION
            # --------------------------------------------------

            prediction = self.model.predict(X)[0]

            # --------------------------------------------------
            # PROBABILITY
            # --------------------------------------------------

            probability = None
            class_probabilities = None

            if hasattr(self.model, "predict_proba"):

                probabilities = self.model.predict_proba(X)[0]

                probability = float(
                    np.max(probabilities)
                )

                class_probabilities = [
                    float(x)
                    for x in probabilities
                ]

            # --------------------------------------------------
            # RESULT
            # --------------------------------------------------

            return {
                "agent": self.name,
                "model": self.model_name,
                "status": "completed",
                "prediction": str(prediction),
                "probability": probability,
                "class_probabilities": class_probabilities
            }

        except Exception as e:

            return {
                "agent": self.name,
                "model": self.model_name,
                "status": "error",
                "prediction": None,
                "probability": None,
                "error": str(e)
            }
