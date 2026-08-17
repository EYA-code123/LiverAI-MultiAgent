%%writefile /content/LiverAI-MultiAgent/agents/cirrhosis_agent.py

import pandas as pd
import numpy as np


class CirrhosisAgent:

    def __init__(self, package):

        self.name = "CirrhosisAgent"
        self.model_name = "XGBoost"

        # ==================================================
        # PACKAGE
        # ==================================================

        self.package = package

        self.model = package["model"]

        self.feature_names = package["feature_names"]

        self.numerical_columns = package[
            "numerical_columns"
        ]

        self.categorical_columns = package[
            "categorical_columns"
        ]

        self.encoders = package["encoders"]

        self.target_encoder = package[
            "target_encoder"
        ]

        self.numerical_imputer = package[
            "numerical_imputer"
        ]

        self.categorical_imputer = package[
            "categorical_imputer"
        ]

    # ======================================================
    # PREDICT
    # ======================================================

    def predict(self, patient_data):

        # --------------------------------------------------
        # Convert input to DataFrame
        # --------------------------------------------------

        if isinstance(patient_data, dict):

            X = pd.DataFrame(
                [patient_data]
            )

        elif isinstance(patient_data, pd.DataFrame):

            X = patient_data.copy()

        else:

            X = pd.DataFrame(
                [patient_data],
                columns=self.feature_names
            )

        # --------------------------------------------------
        # Check missing features
        # --------------------------------------------------

        missing_features = [
            col
            for col in self.feature_names
            if col not in X.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing cirrhosis features: "
                + str(missing_features)
            )

        # --------------------------------------------------
        # Keep ONLY model features
        # --------------------------------------------------

        X = X[
            self.feature_names
        ].copy()

        # ==================================================
        # NUMERICAL FEATURES
        # ==================================================

        numerical_features = [
            col
            for col in self.numerical_columns
            if col in self.feature_names
        ]

        if numerical_features:

            # ----------------------------------------------
            # Convert numerical values
            # ----------------------------------------------

            for col in numerical_features:

                X[col] = pd.to_numeric(
                    X[col],
                    errors="coerce"
                )

            # ----------------------------------------------
            # IMPORTANT
            #
            # The saved imputer was fitted with Stage.
            # Therefore we cannot directly call:
            #
            # numerical_imputer.transform(X[numerical_features])
            #
            # Instead, use the saved statistics only for
            # the features that belong to the model.
            # ----------------------------------------------

            imputer_features = [
                col
                for col in self.numerical_columns
                if col in self.feature_names
            ]

            statistics = (
                self.numerical_imputer.statistics_
            )

            # Mapping:
            # column -> saved median

            median_map = dict(
                zip(
                    self.numerical_columns,
                    statistics
                )
            )

            for col in numerical_features:

                median = median_map.get(
                    col,
                    np.nan
                )

                X[col] = X[col].fillna(
                    median
                )

        # ==================================================
        # CATEGORICAL FEATURES
        # ==================================================

        categorical_features = [
            col
            for col in self.categorical_columns
            if col in self.feature_names
        ]

        if categorical_features:

            # ----------------------------------------------
            # Fill missing categorical values manually
            # using the saved imputer statistics.
            # ----------------------------------------------

            categorical_statistics = (
                self.categorical_imputer.statistics_
            )

            categorical_medians = dict(
                zip(
                    self.categorical_columns,
                    categorical_statistics
                )
            )

            for col in categorical_features:

                fill_value = (
                    categorical_medians.get(
                        col,
                        None
                    )
                )

                if fill_value is not None:

                    X[col] = X[col].fillna(
                        fill_value
                    )

        # ==================================================
        # ENCODE CATEGORICAL FEATURES
        # ==================================================

        for col in categorical_features:

            if col not in self.encoders:
                continue

            encoder = self.encoders[col]

            values = X[col].astype(str)

            known_values = set(
                encoder.classes_
            )

            # ----------------------------------------------
            # Unknown categories
            # ----------------------------------------------

            fallback = encoder.classes_[0]

            values = values.apply(
                lambda x:
                x if x in known_values
                else fallback
            )

            X[col] = encoder.transform(
                values
            )

        # ==================================================
        # EXACT FEATURE ORDER
        # ==================================================

        X = X[
            self.feature_names
        ]

        # ==================================================
        # PREDICTION
        # ==================================================

        prediction_encoded = (
            self.model.predict(X)[0]
        )

        # ==================================================
        # PROBABILITY
        # ==================================================

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

        # ==================================================
        # DECODE TARGET
        # ==================================================

        try:

            prediction = (
                self.target_encoder
                .inverse_transform(
                    [prediction_encoded]
                )[0]
            )

        except Exception:

            prediction = prediction_encoded

        # ==================================================
        # RESULT
        # ==================================================

        return {

            "agent": self.name,

            "model": self.model_name,

            "prediction": str(
                prediction
            ),

            "probability": probability,

            "status": "completed"
        }
