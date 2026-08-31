import numpy as np
import pandas as pd


class FattyLiverAgent:

    def __init__(self, model_package):

        self.name = "FattyLiverAgent"

        self.model_name = model_package.get(
            "model_name",
            "LightGBM"
        )

        # ------------------------------------------------------
        # MODEL
        # ------------------------------------------------------

        self.model = model_package["model"]

        # ------------------------------------------------------
        # FEATURES
        # ------------------------------------------------------

        self.feature_names = list(
            model_package.get(
                "feature_names",
                []
            )
        )

        # ------------------------------------------------------
        # NUMERICAL FEATURES
        # ------------------------------------------------------

        self.numerical_columns = list(
            model_package.get(
                "numerical_columns",
                []
            )
        )

        # ------------------------------------------------------
        # CATEGORICAL FEATURES
        # ------------------------------------------------------

        self.categorical_columns = list(
            model_package.get(
                "categorical_columns",
                []
            )
        )

        # ------------------------------------------------------
        # ENCODERS
        # ------------------------------------------------------

        self.encoders = model_package.get(
            "encoders",
            {}
        )

        # ------------------------------------------------------
        # IMPUTERS
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # TARGET
        # ------------------------------------------------------

        self.target_name = model_package.get(
            "target_name",
            "status"
        )

        self.target_classes = model_package.get(
            "target_classes",
            [0, 1]
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

            # --------------------------------------------------
            # CREATE DATAFRAME
            # --------------------------------------------------

            X = self._create_dataframe(
                patient_data
            )

            # --------------------------------------------------
            # REMOVE TARGET IF PRESENT
            # --------------------------------------------------

            if self.target_name in X.columns:

                X = X.drop(
                    columns=[self.target_name]
                )

            # --------------------------------------------------
            # ADD MISSING FEATURES
            # --------------------------------------------------

            for col in self.feature_names:

                if col not in X.columns:

                    X[col] = np.nan

            # --------------------------------------------------
            # KEEP ONLY MODEL FEATURES
            # --------------------------------------------------

            X = X[
                self.feature_names
            ].copy()

            # --------------------------------------------------
            # NUMERICAL IMPUTATION
            # --------------------------------------------------

            numerical_features = [
                c
                for c in self.numerical_columns
                if c in X.columns
            ]

            if (
                self.numerical_imputer is not None
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
                self.categorical_imputer is not None
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
            # FINAL FEATURE ORDER
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

            prediction_encoded = int(
                prediction_encoded
            )

            # --------------------------------------------------
            # PROBABILITIES
            # --------------------------------------------------

            probabilities = (
                self.model.predict_proba(X)[0]
            )

            probabilities = np.asarray(
                probabilities,
                dtype=float
            )

            confidence = float(
                probabilities.max()
            )

            uncertainty = float(
                1.0 - confidence
            )

            # --------------------------------------------------
            # PREDICTED LABEL
            # --------------------------------------------------

            if (
                prediction_encoded
                < len(self.target_classes)
            ):

                predicted_label = (
                    self.target_classes[
                        prediction_encoded
                    ]
                )

            else:

                predicted_label = (
                    prediction_encoded
                )

            # --------------------------------------------------
            # RESULT
            # --------------------------------------------------

            return {

                "agent":
                    self.name,

                "model":
                    self.model_name,

                "prediction":
                    str(predicted_label),

                "probability":
                    float(confidence),

                "confidence":
                    float(confidence),

                "uncertainty":
                    float(uncertainty),

                "class_probabilities":
                    [
                        float(x)
                        for x in probabilities
                    ],

                "quality":
                    1.0,

                "details":
                    {
                        "task_type":
                            "classification",

                        "disease":
                            "fatty_liver",

                        "target":
                            self.target_name,

                        "classes":
                            [
                                str(x)
                                for x
                                in self.target_classes
                            ],

                        "features":
                            self.feature_names
                    },

                "status":
                    "completed",

                "error":
                    None
            }

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

                "confidence":
                    None,

                "uncertainty":
                    None,

                "class_probabilities":
                    None,

                "quality":
                    0.0,

                "details":
                    {},

                "status":
                    "error",

                "error":
                    str(e)
            }
