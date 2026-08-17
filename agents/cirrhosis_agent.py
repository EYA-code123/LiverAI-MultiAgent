%%writefile /content/LiverAI-MultiAgent/agents/cirrhosis_agent.py

import pandas as pd
import numpy as np


class CirrhosisAgent:

    def __init__(self, package):

        self.name = "CirrhosisAgent"
        self.model_name = "XGBoost"

        # ==================================================
        # LOAD PACKAGE
        # ==================================================

        self.package = package

        self.model = package["model"]

        self.feature_names = list(
            package["feature_names"]
        )

        self.numerical_columns = list(
            package["numerical_columns"]
        )

        self.categorical_columns = list(
            package["categorical_columns"]
        )

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

        # ==================================================
        # IMPORTANT
        # ==================================================
        # The saved numerical imputer expects Stage,
        # but the XGBoost model does NOT use Stage.
        #
        # Therefore:
        #
        # 1. Imputer receives its original 12 columns
        # 2. Stage is removed before XGBoost prediction
        # ==================================================

        if hasattr(
            self.numerical_imputer,
            "feature_names_in_"
        ):

            self.imputer_numerical_features = list(
                self.numerical_imputer.feature_names_in_
            )

        else:

            self.imputer_numerical_features = [
                col
                for col in self.numerical_columns
            ]

    # ======================================================
    # PREDICT
    # ======================================================

    def predict(self, patient_data):

        # ==================================================
        # CONVERT INPUT TO DATAFRAME
        # ==================================================

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
                columns=self.feature_names
            )

        # ==================================================
        # ADD MISSING INPUT FEATURES
        # ==================================================

        for col in self.feature_names:

            if col not in X.columns:

                X[col] = np.nan

        # ==================================================
        # NUMERICAL IMPUTATION
        # ==================================================

        # The saved imputer expects 12 columns,
        # including Stage.

        imputer_features = (
            self.imputer_numerical_features
        )

        X_imputer = pd.DataFrame(
            index=X.index
        )

        for col in imputer_features:

            if col in X.columns:

                X_imputer[col] = X[col]

            else:

                # Stage is not a model feature.
                # Give it NaN so the saved imputer
                # can use its learned median.
                X_imputer[col] = np.nan

        X_imputer = (
            self.numerical_imputer
            .transform(X_imputer)
        )

        X_imputer = pd.DataFrame(
            X_imputer,
            columns=imputer_features,
            index=X.index
        )

        # ==================================================
        # KEEP ONLY MODEL NUMERICAL FEATURES
        # ==================================================

        for col in self.feature_names:

            if col in imputer_features:

                X[col] = X_imputer[col]

        # ==================================================
        # CATEGORICAL IMPUTATION
        # ==================================================

        categorical_features = [

            col

            for col in self.categorical_columns

            if col in self.feature_names

        ]

        if categorical_features:

            X_cat = X[
                categorical_features
            ].copy()

            X_cat = (
                self.categorical_imputer
                .transform(X_cat)
            )

            X_cat = pd.DataFrame(
                X_cat,
                columns=categorical_features,
                index=X.index
            )

            X[
                categorical_features
            ] = X_cat

        # ==================================================
        # ENCODE CATEGORICAL FEATURES
        # ==================================================

        for col in categorical_features:

            if col not in self.encoders:
                continue

            encoder = self.encoders[col]

            values = (
                X[col]
                .astype(str)
            )

            known_values = set(
                encoder.classes_
            )

            values = values.apply(

                lambda value:

                value
                if value in known_values
                else encoder.classes_[0]

            )

            X[col] = (
                encoder.transform(
                    values
                )
            )

        # ==================================================
        # FINAL MODEL INPUT
        # ==================================================

        # IMPORTANT:
        # Use ONLY the 18 features expected
        # by XGBoost.

        X_model = X[
            self.feature_names
        ].copy()

        # Force exact order
        X_model = X_model[
            self.feature_names
        ]

        # ==================================================
        # PREDICTION
        # ==================================================

        prediction_encoded = (
            self.model
            .predict(X_model)[0]
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
                self.model
                .predict_proba(
                    X_model
                )[0]
            )

            probability = float(
                np.max(
                    probabilities
                )
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

            prediction = (
                prediction_encoded
            )

        # ==================================================
        # RESULT
        # ==================================================

        return {

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
