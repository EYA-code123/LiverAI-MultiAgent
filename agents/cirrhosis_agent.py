
"""
Cirrhosis Agent
Uses the trained XGBoost model and the preprocessing
information saved in XGBoost_Cirrhosis.pkl.
"""

import numpy as np
import pandas as pd


class CirrhosisAgent:

    def __init__(self, artifact):
        """
        artifact = dictionary loaded from XGBoost_Cirrhosis.pkl
        """

        self.name = "CirrhosisAgent"
        self.model_name = "XGBoost"

        self.model = artifact["model"]

        self.feature_names = artifact["feature_names"]

        self.numerical_columns = artifact["numerical_columns"]

        self.categorical_columns = artifact["categorical_columns"]

        self.encoders = artifact["encoders"]

        self.target_encoder = artifact["target_encoder"]

        self.numerical_imputer = artifact["numerical_imputer"]

        self.categorical_imputer = artifact["categorical_imputer"]

    def predict(self, patient_data):
        """
        Predict cirrhosis stage.

        patient_data must be a dictionary containing
        the 18 features expected by the model.
        """

        # --------------------------------------------------
        # 1. Convert input to DataFrame
        # --------------------------------------------------

        if isinstance(patient_data, dict):

            df = pd.DataFrame([patient_data])

        elif isinstance(patient_data, pd.DataFrame):

            df = patient_data.copy()

        else:

            raise TypeError(
                "patient_data must be a dictionary "
                "or pandas DataFrame."
            )

        # --------------------------------------------------
        # 2. Check required features
        # --------------------------------------------------

        missing_features = [
            feature
            for feature in self.feature_names
            if feature not in df.columns
        ]

        if missing_features:

            raise ValueError(
                f"Missing features: {missing_features}"
            )

        # Keep exactly the training order
        df = df[self.feature_names].copy()

        # --------------------------------------------------
        # 3. Numerical imputation
        # --------------------------------------------------

        if self.numerical_columns:

            df[self.numerical_columns] = (
                self.numerical_imputer.transform(
                    df[self.numerical_columns]
                )
            )

        # --------------------------------------------------
        # 4. Categorical imputation
        # --------------------------------------------------

        if self.categorical_columns:

            df[self.categorical_columns] = (
                self.categorical_imputer.transform(
                    df[self.categorical_columns]
                )
            )

        # --------------------------------------------------
        # 5. Encode categorical variables
        # --------------------------------------------------

        for column in self.categorical_columns:

            encoder = self.encoders[column]

            df[column] = df[column].astype(str)

            df[column] = encoder.transform(
                df[column]
            )

        # --------------------------------------------------
        # 6. XGBoost prediction
        # --------------------------------------------------

        prediction = self.model.predict(df)[0]

        # --------------------------------------------------
        # 7. Prediction probability
        # --------------------------------------------------

        probability = None

        if hasattr(self.model, "predict_proba"):

            probabilities = (
                self.model.predict_proba(df)[0]
            )

            probability = float(
                np.max(probabilities)
            )

        # --------------------------------------------------
        # 8. Convert encoded class to original class
        # --------------------------------------------------

        predicted_stage = (
            self.target_encoder.inverse_transform(
                [prediction]
            )[0]
        )

        # --------------------------------------------------
        # 9. Return structured result
        # --------------------------------------------------

        return {
            "agent": self.name,
            "model": self.model_name,
            "prediction": str(predicted_stage),
            "probability": probability,
            "status": "completed"
        }
