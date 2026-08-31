import numpy as np
import pandas as pd


class FattyLiverAgent:

    def __init__(self, model_package):

        self.name = "FattyLiverAgent"

        # ============================================================
        # MODEL
        # ============================================================

        self.model_name = model_package.get(
            "model_name",
            "LightGBM"
        )

        self.model = model_package["model"]

        # ============================================================
        # FEATURES
        # ============================================================

        self.feature_names = list(
            model_package.get(
                "feature_names",
                []
            )
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

        # ============================================================
        # PREPROCESSING
        # ============================================================

        self.encoders = model_package.get(
            "encoders",
            {}
        )

        self.numerical_imputer = (
            model_package.get(
                "numerical_imputer",
                None
            )
        )

        # ============================================================
        # TARGET
        # ============================================================

        self.target_name = model_package.get(
            "target_name",
            "status"
        )

        self.target_classes = model_package.get(
            "target_classes",
            ["0", "1"]
        )

    # ================================================================
    # CREATE DATAFRAME
    # ================================================================

    def _create_dataframe(self, patient_data):

        if isinstance(patient_data, dict):

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

    # ================================================================
    # PREDICT
    # ================================================================

    def predict(self, patient_data):

        try:

            X = self._create_dataframe(
                patient_data
            )

            # ========================================================
            # REMOVE TARGET
            # ========================================================

            if self.target_name in X.columns:

                X = X.drop(
                    columns=[self.target_name]
                )

            # ========================================================
            # REMOVE NON-FEATURE COLUMNS
            # ========================================================

            # The model must receive exactly the features
            # used during training.

            for col in list(X.columns):

                if col not in self.feature_names:

                    X = X.drop(
                        columns=[col]
                    )

            # ========================================================
            # ADD MISSING FEATURES
            # ========================================================

            for col in self.feature_names:

                if col not in X.columns:

                    X[col] = np.nan

            # ========================================================
            # FINAL FEATURE ORDER
            # ========================================================

            X = X[
                self.feature_names
            ].copy()

            # ========================================================
            # NUMERICAL IMPUTATION
            # ========================================================

            numerical_features = [
                col
                for col in self.numerical_columns
                if col in X.columns
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

            # ========================================================
            # CATEGORICAL ENCODING
            # ========================================================

            categorical_features = [
                col
                for col in self.categorical_columns
                if col in X.columns
            ]

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

                # Unknown values are replaced
                # by the first known class.

                values = values.apply(
                    lambda value:
                    value
                    if value in known_values
                    else encoder.classes_[0]
                )

                X[col] = encoder.transform(
                    values
                )

            # ========================================================
            # MODEL PREDICTION
            # ========================================================

            prediction_encoded = (
                self.model.predict(X)[0]
            )

            # ========================================================
            # PROBABILITIES
            # ========================================================

            probabilities = None

            if hasattr(
                self.model,
                "predict_proba"
            ):

                probabilities = (
                    self.model.predict_proba(X)[0]
                )

            # ========================================================
            # DECODE PREDICTION
            # ========================================================

            prediction = str(
                prediction_encoded
            )

            try:

                encoded_int = int(
                    prediction_encoded
                )

                if (
                    encoded_int
                    < len(self.target_classes)
                ):

                    prediction = str(
                        self.target_classes[
                            encoded_int
                        ]
                    )

            except Exception:

                prediction = str(
                    prediction_encoded
                )

            # ========================================================
            # CONFIDENCE
            # ========================================================

            if probabilities is not None:

                confidence = float(
                    np.max(probabilities)
                )

                probability_list = [
                    float(x)
                    for x in probabilities
                ]

            else:

                confidence = None

                probability_list = None

            # ========================================================
            # UNCERTAINTY
            # ========================================================

            if confidence is not None:

                uncertainty = float(
                    1.0 - confidence
                )

            else:

                uncertainty = None

            # ========================================================
            # QUALITY
            # ========================================================

            quality = 1.0

            # ========================================================
            # RESULT
            # ========================================================

            return {

                "agent":
                    self.name,

                "model":
                    self.model_name,

                "prediction":
                    prediction,

                "probability":
                    confidence,

                "confidence":
                    confidence,

                "uncertainty":
                    uncertainty,

                "quality":
                    quality,

                "class_probabilities":
                    probability_list,

                "details": {

                    "task_type":
                        "classification",

                    "disease":
                        "fatty_liver",

                    "target":
                        self.target_name,

                    "classes":
                        [
                            str(x)
                            for x in self.target_classes
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

                "quality":
                    0.0,

                "class_probabilities":
                    None,

                "details":
                    {
                        "task_type":
                            "classification",

                        "disease":
                            "fatty_liver"
                    },

                "status":
                    "error",

                "error":
                    str(e)
            }
