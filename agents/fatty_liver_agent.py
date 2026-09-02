# =============================================================================
# Fatty Liver Agent
# =============================================================================

import time
import numpy as np
import pandas as pd


class FattyLiverAgent:

    def __init__(
        self,
        model_package
    ):

        self.name = (
            "FattyLiverAgent"
        )

        self.model_name = (
            model_package.get(
                "model_name",
                "LightGBM"
            )
        )

        self.model = (
            model_package["model"]
        )

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

        self.encoders = (
            model_package.get(
                "encoders",
                {}
            )
        )

        self.numerical_imputer = (
            model_package.get(
                "numerical_imputer"
            )
        )

        self.target_name = (
            model_package.get(
                "target_name",
                "status"
            )
        )

        self.target_classes = list(
            model_package.get(
                "target_classes",
                ["0", "1"]
            )
        )

    # =========================================================================
    # DATAFRAME
    # =========================================================================

    def _create_dataframe(
        self,
        patient_data
    ):

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

    # =========================================================================
    # PREDICT
    # =========================================================================

    def predict(
        self,
        patient_data
    ):

        start_time = (
            time.perf_counter()
        )

        try:

            X = self._create_dataframe(
                patient_data
            )

            # -----------------------------------------------------------------
            # REMOVE TARGET
            # -----------------------------------------------------------------

            if self.target_name in X.columns:

                X = X.drop(
                    columns=[
                        self.target_name
                    ]
                )

            # -----------------------------------------------------------------
            # REMOVE UNKNOWN COLUMNS
            # -----------------------------------------------------------------

            if self.feature_names:

                X = X[
                    [
                        col

                        for col
                        in X.columns

                        if col
                        in self.feature_names
                    ]
                ]

                # -------------------------------------------------------------
                # ADD MISSING FEATURES
                # -------------------------------------------------------------

                missing_features = []

                for col in self.feature_names:

                    if col not in X.columns:

                        X[col] = np.nan

                        missing_features.append(
                            col
                        )

            else:

                missing_features = []

            # -----------------------------------------------------------------
            # ORDER
            # -----------------------------------------------------------------

            if self.feature_names:

                X = X[
                    self.feature_names
                ].copy()

            # -----------------------------------------------------------------
            # MISSING RATIO
            # -----------------------------------------------------------------

            missing_ratio = float(
                X.isna()
                .sum()
                .sum()
                /
                max(
                    X.shape[1],
                    1
                )
            )

            # -----------------------------------------------------------------
            # NUMERICAL IMPUTATION
            # -----------------------------------------------------------------

            numerical_features = [

                col

                for col in
                self.numerical_columns

                if col in X.columns
            ]

            if (
                self.numerical_imputer
                is not None
                and numerical_features
            ):

                X[
                    numerical_features
                ] = (
                    self.numerical_imputer
                    .transform(
                        X[
                            numerical_features
                        ]
                    )
                )

            # -----------------------------------------------------------------
            # CATEGORICAL ENCODING
            # -----------------------------------------------------------------

            categorical_features = [

                col

                for col in
                self.categorical_columns

                if col in X.columns
            ]

            for col in categorical_features:

                if col not in self.encoders:

                    continue

                encoder = (
                    self.encoders[col]
                )

                values = (
                    X[col].astype(str)
                )

                known_values = set(
                    encoder.classes_
                )

                values = values.apply(

                    lambda value:

                    value

                    if value
                    in known_values

                    else
                    encoder.classes_[0]
                )

                X[col] = (
                    encoder.transform(
                        values
                    )
                )

            # -----------------------------------------------------------------
            # PREDICT
            # -----------------------------------------------------------------

            prediction_encoded = (
                self.model.predict(X)[0]
            )

            # -----------------------------------------------------------------
            # PROBABILITIES
            # -----------------------------------------------------------------

            probabilities = None

            if hasattr(
                self.model,
                "predict_proba"
            ):

                probabilities = (
                    self.model
                    .predict_proba(X)[0]
                )

            if probabilities is not None:

                confidence = float(
                    np.max(
                        probabilities
                    )
                )

                class_probabilities = [

                    float(x)

                    for x in probabilities
                ]

            else:

                confidence = 0.0

                class_probabilities = None

            uncertainty = (
                1.0 - confidence
            )

            quality = max(
                0.0,
                1.0 - missing_ratio
            )

            # -----------------------------------------------------------------
            # DECODE
            # -----------------------------------------------------------------

            try:

                encoded_int = int(
                    prediction_encoded
                )

                if (
                    0
                    <= encoded_int
                    <
                    len(
                        self.target_classes
                    )
                ):

                    prediction = str(
                        self.target_classes[
                            encoded_int
                        ]
                    )

                else:

                    prediction = str(
                        prediction_encoded
                    )

            except Exception:

                prediction = str(
                    prediction_encoded
                )

            latency_ms = (
                time.perf_counter()
                -
                start_time
            ) * 1000.0

            return {

                "agent":
                    self.name,

                "task_type":
                    "fatty_liver_classification",

                "model":
                    self.model_name,

                "status":
                    "success",

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

                "missing_data_ratio":
                    missing_ratio,

                "latency_ms":
                    latency_ms,

                "class_probabilities":
                    class_probabilities,

                "details": {

                    "task_type":
                        "fatty_liver_classification",

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
                        self.feature_names,

                    "missing_features":
                        missing_features
                },

                "error":
                    None
            }

        except Exception as e:

            latency_ms = (
                time.perf_counter()
                -
                start_time
            ) * 1000.0

            return {

                "agent":
                    self.name,

                "task_type":
                    "fatty_liver_classification",

                "model":
                    self.model_name,

                "status":
                    "error",

                "prediction":
                    None,

                "probability":
                    None,

                "confidence":
                    0.0,

                "uncertainty":
                    1.0,

                "quality":
                    0.0,

                "missing_data_ratio":
                    1.0,

                "latency_ms":
                    latency_ms,

                "details": {

                    "task_type":
                        "fatty_liver_classification",

                    "disease":
                        "fatty_liver"
                },

                "error":
                    str(e)
            }
