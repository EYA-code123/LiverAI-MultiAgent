%%writefile /content/LiverAI-MultiAgent/agents/fatty_liver_agent.py

"""
Fatty Liver Agent
Uses the trained LightGBM model on the BUPA Liver Disorders dataset.
"""

import os
import numpy as np
import pandas as pd
import joblib


class FattyLiverAgent:

    def __init__(self, model=None):

        self.name = "FattyLiverAgent"
        self.model_name = "LightGBM"

        # ==========================================================
        # Load trained model automatically
        # ==========================================================

        if model is not None:

            self.model = model

        else:

            model_path = (
                "/content/LiverAI-MultiAgent/"
                "02_Fatty_Liver_Classification/"
                "models/"
                "FattyLiver_LightGBM.pkl"
            )

            if not os.path.exists(model_path):

                raise FileNotFoundError(
                    f"Fatty Liver model not found:\n{model_path}"
                )

            self.model = joblib.load(model_path)

        # BUPA model features
        self.features = [
            "mcv",
            "alkphos",
            "sgpt",
            "sgot",
            "gammagt",
            "drinks"
        ]


    # ==========================================================
    # Prediction
    # ==========================================================

    def predict(self, patient_data):

        # ------------------------------------------------------
        # Convert input to DataFrame
        # ------------------------------------------------------

        if isinstance(patient_data, dict):

            missing = [
                feature
                for feature in self.features
                if feature not in patient_data
            ]

            if missing:

                raise ValueError(
                    f"Missing features: {missing}"
                )

            X = pd.DataFrame(
                [[patient_data[feature] for feature in self.features]],
                columns=self.features
            )

        else:

            X = np.asarray(patient_data)

            if X.ndim == 1:
                X = X.reshape(1, -1)

            if X.shape[1] != len(self.features):

                raise ValueError(
                    f"Expected {len(self.features)} features "
                    f"{self.features}, "
                    f"but received {X.shape[1]}"
                )

            X = pd.DataFrame(
                X,
                columns=self.features
            )

        # ------------------------------------------------------
        # Prediction
        # ------------------------------------------------------

        prediction = self.model.predict(X)[0]

        # ------------------------------------------------------
        # Probability
        # ------------------------------------------------------

        probability = None

        if hasattr(self.model, "predict_proba"):

            probabilities = self.model.predict_proba(X)[0]

            probability = float(
                np.max(probabilities)
            )

        # ------------------------------------------------------
        # Interpretation
        # ------------------------------------------------------

        prediction_str = str(prediction)

        if prediction_str == "1":

            interpretation = (
                "Class 1 predicted by the BUPA model"
            )

        elif prediction_str == "2":

            interpretation = (
                "Class 2 predicted by the BUPA model"
            )

        else:

            interpretation = (
                f"Predicted class: {prediction_str}"
            )

        # ------------------------------------------------------
        # Return result
        # ------------------------------------------------------

        return {

            "agent": self.name,

            "model": self.model_name,

            "prediction": prediction_str,

            "probability": probability,

            "interpretation": interpretation,

            "status": "completed"

        }
