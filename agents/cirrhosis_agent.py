"""
Cirrhosis Agent
Uses the trained XGBoost model and the preprocessing
information saved in XGBoost_Cirrhosis.pkl.
"""

import numpy as np
import pandas as pd


class CirrhosisAgent:

    def __init__(self, artifact):

        self.name = "CirrhosisAgent"
        self.model_name = "XGBoost"

        self.model = artifact["model"]

        self.feature_names = artifact["feature_names"]

        self.numerical_columns = artifact["numerical_columns"]

        self.categorical_columns = artifact["categorical_columns"]

        self.encoders = artifact["encoders"]

        self.target_encoder = artifact["target_encoder"]

        self.numerical_imputer = artifact.get(
            "numerical_imputer"
        )

        self.categorical_imputer = artifact.get(
            "categorical_imputer"
        )

    def predict(self, patient_data):

        # ==========================================
        # 1. Convert input to DataFrame
        # ==========================================

        if isinstance(patient_data, dict):

            df = pd.DataFrame([patient_data])

        elif isinstance(patient_data, pd.DataFrame):

            df = patient_data.copy()

        else:

            raise TypeError(
                "patient_data must be a dictionary "
                "or pandas DataFrame."
            )

        # ==========================================
        # 2. Check features
        # ==========================================

        missing_features = [
            feature
            for feature in self.feature_names
            if feature not in df.columns
        ]

        if missing_features:

            raise ValueError(
                f"Missing features: {missing_features}"
            )

        # Keep only model features
        df = df[self.feature_names].copy()

        # ==========================================
        # 3. Numerical preprocessing
        # ==========================================

        # Remove target column if it was accidentally
        # stored inside numerical_columns
        numerical_columns = [
            column
            for column in self.numerical_columns
            if column in self.feature_names
        ]

        if numerical_columns:

            # Fill missing numerical values
            # using the statistics stored by the imputer
            if self.numerical_imputer is not None:

                for column in numerical_columns:

                    if df[column].isna().any():

                        if column in self.numerical_columns:

                            index = self.numerical_columns.index(
                                column
                            )

                            median_value = (
                                self.numerical_imputer.statistics_[
                                    index
                                ]
                            )

                            df[column] = df[column].fillna(
                                median_value
                            )

        # ==========================================
        # 4. Categorical preprocessing
        # ==========================================

        categorical_columns = [
            column
            for column in self.categorical_columns
            if column in self.feature_names
        ]

        if categorical_columns:

            for column in categorical_columns:

                if df[column].isna().any():

                    if self.categorical_imputer is not None:

                        index = self.categorical_columns.index(
                            column
                        )

                        most_frequent = (
                            self.categorical_imputer.statistics_[
                                index
                            ]
                        )

                        df[column] = df[column].fillna(
                            most_frequent
                        )

        # ==========================================
        # 5. Encode categorical variables
        # ==========================================

        for column in categorical_columns:

            encoder = self.encoders[column]

            df[column] = df[column].astype(str)

            unknown_values = set(
                df[column]
            ) - set(encoder.classes_)

            if unknown_values:

                raise ValueError(
                    f"Unknown value(s) in {column}: "
                    f"{unknown_values}"
                )

            df[column] = encoder.transform(
                df[column]
            )

        # ==========================================
        # 6. Prediction
        # ==========================================

        prediction = self.model.predict(df)[0]

        # ==========================================
        # 7. Probability
        # ==========================================

        probability = None

        if hasattr(self.model, "predict_proba"):

            probabilities = (
                self.model.predict_proba(df)[0]
            )

            probability = float(
                np.max(probabilities)
            )

        # ==========================================
        # 8. Decode prediction
        # ==========================================

        predicted_stage = (
            self.target_encoder.inverse_transform(
                [prediction]
            )[0]
        )

        # ==========================================
        # 9. Return result
        # ==========================================

        return {
            "agent": self.name,
            "model": self.model_name,
            "prediction": str(predicted_stage),
            "probability": probability,
            "status": "completed"
        }
