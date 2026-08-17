"""
Cirrhosis Prediction Agent
Uses the trained XGBoost model and preprocessing information.
"""

import pandas as pd
import numpy as np


class CirrhosisAgent:

    def __init__(self, model_package):

        self.name = "CirrhosisAgent"
        self.model_name = "XGBoost"

        # Complete saved package
        self.package = model_package

        # XGBoost model
        self.model = model_package["model"]

        # Saved preprocessing information
        self.feature_names = model_package["feature_names"]
        self.numerical_columns = model_package["numerical_columns"]
        self.categorical_columns = model_package["categorical_columns"]

        self.encoders = model_package["encoders"]

        self.target_encoder = model_package["target_encoder"]

        self.numerical_imputer = model_package["numerical_imputer"]
        self.categorical_imputer = model_package["categorical_imputer"]


    def predict(self, patient_data):

        # --------------------------------------------------
        # Convert patient data to DataFrame
        # --------------------------------------------------

        if isinstance(patient_data, dict):

            X = pd.DataFrame([patient_data])

        elif isinstance(patient_data, pd.DataFrame):

            X = patient_data.copy()

        else:

            X = pd.DataFrame(
                [patient_data],
                columns=self.feature_names
            )


        # --------------------------------------------------
        # Keep only model features
        # --------------------------------------------------

        X = X[self.feature_names].copy()


        # --------------------------------------------------
        # Numerical preprocessing
        # --------------------------------------------------

        numerical_features = [
            col for col in self.numerical_columns
            if col in self.feature_names
        ]

        if numerical_features:

            X[numerical_features] = (
                self.numerical_imputer
                .transform(X[numerical_features])
            )


        # --------------------------------------------------
        # Categorical preprocessing
        # --------------------------------------------------

        categorical_features = [
            col for col in self.categorical_columns
            if col in self.feature_names
        ]

        if categorical_features:

            X[categorical_features] = (
                self.categorical_imputer
                .transform(X[categorical_features])
            )


        # --------------------------------------------------
        # Apply saved encoders
        # --------------------------------------------------

        for col in categorical_features:

            if col in self.encoders:

                encoder = self.encoders[col]

                values = X[col].astype(str)

                # Handle unknown categories
                known_values = set(
                    encoder.classes_
                )

                values = values.apply(
                    lambda x:
                    x if x in known_values
                    else encoder.classes_[0]
                )

                X[col] = encoder.transform(values)


        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        prediction_encoded = self.model.predict(X)[0]


        # --------------------------------------------------
        # Probability
        # --------------------------------------------------

        probability = None

        if hasattr(
            self.model,
            "predict_proba"
        ):

            probabilities = (
                self.model
                .predict_proba(X)[0]
            )

            probability = float(
                np.max(probabilities)
            )


        # --------------------------------------------------
        # Decode prediction
        # --------------------------------------------------

        try:

            prediction = (
                self.target_encoder
                .inverse_transform(
                    [prediction_encoded]
                )[0]
            )

        except Exception:

            prediction = prediction_encoded


        # --------------------------------------------------
        # Return result
        # --------------------------------------------------

        return {

            "agent": self.name,

            "model": self.model_name,

            "prediction": str(
                prediction
            ),

            "probability": probability,

            "status": "completed"

        }
