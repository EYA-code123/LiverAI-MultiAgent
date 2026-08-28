import numpy as np
import pandas as pd


class FattyLiverAgent:

    def __init__(self, model_package):

        self.name = "FattyLiverAgent"

        self.model_name = model_package.get(
            "model_name",
            "LightGBM"
        )

        self.model = model_package["model"]

        self.features = model_package.get(
            "feature_names",
            [
                "mcv",
                "alkphos",
                "sgpt",
                "sgot",
                "gammagt",
                "drinks"
            ]
        )

    # ==========================================================
    # PREDICT
    # ==========================================================

    def predict(self, patient_data):

        try:

            # --------------------------------------------------
            # DATAFRAME
            # --------------------------------------------------

            if isinstance(
                patient_data,
                dict
            ):

                X = pd.DataFrame(
                    [patient_data]
                )

            elif isinstance(
                patient_data,
                pd.DataFrame
            ):

                X = patient_data.copy()

            else:

                X = pd.DataFrame(
                    [patient_data],
                    columns=self.features
                )

            # --------------------------------------------------
            # MISSING FEATURES
            # --------------------------------------------------

            for feature in self.features:

                if feature not in X.columns:

                    X[feature] = np.nan

            # --------------------------------------------------
            # EXACT FEATURE ORDER
            # --------------------------------------------------

            X = X[
                self.features
            ].copy()

            # --------------------------------------------------
            # PREDICTION
            # --------------------------------------------------

            prediction = self.model.predict(
                X
            )[0]

            # --------------------------------------------------
            # PROBABILITY
            # --------------------------------------------------

            probability = None
            probabilities = None

            if hasattr(
                self.model,
                "predict_proba"
            ):

                probabilities = (
                    self.model.predict_proba(X)[0]
                )

                probability = float(
                    np.max(probabilities)
                )

            # --------------------------------------------------
            # RESULT
            # --------------------------------------------------

            result = {

                "agent":
                    self.name,

                "model":
                    self.model_name,

                "prediction":
                    str(prediction),

                "probability":
                    probability,

                "status":
                    "completed"
            }

            if probabilities is not None:

                result[
                    "class_probabilities"
                ] = [
                    float(x)
                    for x in probabilities
                ]

            return result

        except Exception as e:

            return {

                "agent":
                    self.name,

                "model":
                    self.model_name,

                "prediction":
                    None,

                "probability":
                    None,

                "status":
                    "error",

                "error":
                    str(e)
            }
