import numpy as np
import pandas as pd


class CirrhosisAgent:

    def __init__(self, model_package):

        self.name = "CirrhosisAgent"

        self.model_name = model_package.get(
            "model_name",
            "XGBoost"
        )

        self.model = model_package["model"]

        self.feature_names = list(
            model_package["feature_names"]
        )

        self.numerical_columns = list(
            model_package.get(
                "numerical_columns",
                []
            )
        )

        self.categorical_columns = list(
            model_package.get(
                "categorical_columns",
                []
            )
        )

        self.encoders = model_package.get(
            "encoders",
            {}
        )

        self.target_encoder = model_package.get(
            "target_encoder",
            None
        )

        self.numerical_imputer = (
            model_package.get(
                "numerical_imputer",
                None
            )
        )

        self.categorical_imputer = (
            model_package.get(
                "categorical_imputer",
                None
            )
        )

    # ==========================================================
    # CREATE DATAFRAME
    # ==========================================================

    def _create_dataframe(self, patient_data):

        if isinstance(
            patient_data,
            dict
        ):

            return pd.DataFrame(
                [patient_data]
            )

        if isinstance(
            patient_data,
            pd.DataFrame
        ):

            return patient_data.copy()

        return pd.DataFrame(
            [patient_data],
            columns=self.feature_names
        )

    # ==========================================================
    # PREDICT
    # ==========================================================

    def predict(self, patient_data):

        try:

            X = self._create_dataframe(
                patient_data
            )

            # --------------------------------------------------
            # ADD REQUIRED FEATURES
            # --------------------------------------------------

            for col in self.feature_names:

                if col not in X.columns:

                    X[col] = np.nan

            # --------------------------------------------------
            # NUMERICAL IMPUTATION
            # --------------------------------------------------

            numerical_features = [
                c
                for c in self.numerical_columns
                if c in X.columns
            ]

            if (
                self.numerical_imputer
                is not None
                and numerical_features
            ):

                X[numerical_features] = (
                    self.numerical_imputer.transform(
                        X[numerical_features]
                    )
                )

            # --------------------------------------------------
            # CATEGORICAL IMPUTATION
            # --------------------------------------------------

            categorical_features = [
                c
                for c in self.categorical_columns
                if c in X.columns
            ]

            if (
                self.categorical_imputer
                is not None
                and categorical_features
            ):

                X[categorical_features] = (
                    self.categorical_imputer.transform(
                        X[categorical_features]
                    )
                )

            # --------------------------------------------------
            # ENCODING
            # --------------------------------------------------

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

                X[col] = encoder.transform(
                    values
                )

            # --------------------------------------------------
            # REMOVE TARGET IF PRESENT
            # --------------------------------------------------

            if "Stage" in X.columns:

                X = X.drop(
                    columns=["Stage"]
                )

            # --------------------------------------------------
            # FINAL ORDER
            # --------------------------------------------------

            X = X[
                self.feature_names
            ].copy()

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
            # DECODE TARGET
            # --------------------------------------------------

            prediction = prediction_encoded

            if self.target_encoder is not None:

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
