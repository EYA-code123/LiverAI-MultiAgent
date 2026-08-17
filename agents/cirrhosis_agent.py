from pathlib import Path

agent_path = "/content/LiverAI-MultiAgent/agents/cirrhosis_agent.py"

code = r'''
import pandas as pd
import numpy as np
import joblib


class CirrhosisAgent:

    def __init__(self, model_package):

        self.name = "CirrhosisAgent"
        self.model_name = "XGBoost"

        # --------------------------------------------------
        # Load package
        # --------------------------------------------------

        self.package = model_package

        self.model = model_package["model"]

        self.feature_names = list(
            model_package["feature_names"]
        )

        self.numerical_columns = list(
            model_package["numerical_columns"]
        )

        self.categorical_columns = list(
            model_package["categorical_columns"]
        )

        self.encoders = model_package["encoders"]

        self.target_encoder = model_package[
            "target_encoder"
        ]

        self.numerical_imputer = model_package[
            "numerical_imputer"
        ]

        self.categorical_imputer = model_package[
            "categorical_imputer"
        ]

        # --------------------------------------------------
        # IMPORTANT FIX
        # --------------------------------------------------

        # The old imputer contains "Stage", but Stage is NOT
        # a feature used by the final XGBoost model.

        self.model_numerical_features = [
            col
            for col in self.feature_names
            if col in self.numerical_columns
        ]

        # Remove Stage automatically
        self.model_numerical_features = [
            col
            for col in self.model_numerical_features
            if col != "Stage"
        ]

        self.model_categorical_features = [
            col
            for col in self.feature_names
            if col in self.categorical_columns
        ]

        print("=" * 70)
        print("CIRRHOSIS AGENT INITIALIZED")
        print("=" * 70)

        print("\nModel features:")
        print(self.feature_names)

        print("\nNumerical features used:")
        print(self.model_numerical_features)

        print("\nCategorical features used:")
        print(self.model_categorical_features)

        print("\nStage removed from prediction pipeline.")

    # ======================================================
    # PREDICT
    # ======================================================

    def predict(self, patient_data):

        # --------------------------------------------------
        # Convert input to DataFrame
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
        # Add missing model features
        # --------------------------------------------------

        for col in self.feature_names:

            if col not in X.columns:

                X[col] = np.nan

        # --------------------------------------------------
        # Keep EXACTLY model features
        # --------------------------------------------------

        X = X[self.feature_names].copy()

        # --------------------------------------------------
        # NUMERICAL PREPROCESSING
        # --------------------------------------------------

        numerical_features = (
            self.model_numerical_features
        )

        if numerical_features:

            # --------------------------------------------------
            # IMPORTANT:
            # The saved imputer incorrectly contains Stage.
            #
            # Therefore we manually use the corresponding
            # statistics for the 11 model numerical features.
            # --------------------------------------------------

            imputer_features = list(
                getattr(
                    self.numerical_imputer,
                    "feature_names_in_",
                    []
                )
            )

            statistics = np.asarray(
                self.numerical_imputer.statistics_
            )

            values = []

            for col in numerical_features:

                if col in imputer_features:

                    index = imputer_features.index(
                        col
                    )

                    statistic = statistics[index]

                else:

                    statistic = np.nan

                values.append(statistic)

            # --------------------------------------------------
            # Fill missing numerical values
            # --------------------------------------------------

            for col, statistic in zip(
                numerical_features,
                values
            ):

                X[col] = pd.to_numeric(
                    X[col],
                    errors="coerce"
                )

                X[col] = X[col].fillna(
                    statistic
                )

        # --------------------------------------------------
        # CATEGORICAL PREPROCESSING
        # --------------------------------------------------

        categorical_features = (
            self.model_categorical_features
        )

        if categorical_features:

            # Use saved categorical imputer
            try:

                X[categorical_features] = (
                    self.categorical_imputer.transform(
                        X[categorical_features]
                    )
                )

            except Exception:

                # Safe fallback
                for col in categorical_features:

                    X[col] = X[col].fillna(
                        "Unknown"
                    )

        # --------------------------------------------------
        # APPLY ENCODERS
        # --------------------------------------------------

        for col in categorical_features:

            if col not in self.encoders:
                continue

            encoder = self.encoders[col]

            values = X[col].astype(str)

            known_values = set(
                encoder.classes_
            )

            # Replace unknown categories
            values = values.apply(
                lambda x:
                x if x in known_values
                else encoder.classes_[0]
            )

            X[col] = encoder.transform(
                values
            )

        # --------------------------------------------------
        # FINAL FEATURE ORDER
        # --------------------------------------------------

        X = X[self.feature_names]

        # Make sure numeric data are numeric
        for col in self.feature_names:

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            )

        # --------------------------------------------------
        # PREDICTION
        # --------------------------------------------------

        prediction_encoded = (
            self.model.predict(X)[0]
        )

        # --------------------------------------------------
        # PROBABILITY
        # --------------------------------------------------

        probability = None

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
        # DECODE TARGET
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
        # RESULT
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
'''

Path(agent_path).write_text(
    code,
    encoding="utf-8"
)

print("✅ cirrhosis_agent.py replaced successfully")
print(agent_path)
