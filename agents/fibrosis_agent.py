# =============================================================================
# Fibrosis Agent
# =============================================================================

import time
import numpy as np
import pandas as pd


class FibrosisAgent:

    def __init__(
        self,
        model
    ):

        self.name = "FibrosisAgent"

        self.model_name = (
            "XGBoost / Random Forest"
        )

        self.model = model

        self.features = [

            "age",
            "male",
            "weight",
            "height",
            "bmi",
            "futime",
            "days",
            "test",
            "value"
        ]

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
            columns=self.features
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

            if patient_data is None:

                raise ValueError(
                    "patient_data is None"
                )

            X = self._create_dataframe(
                patient_data
            )

            # ---------------------------------------------------------------
            # EXPECTED FEATURES
            # ---------------------------------------------------------------

            if hasattr(
                self.model,
                "feature_names_in_"
            ):

                expected_features = list(
                    self.model.feature_names_in_
                )

            else:

                expected_features = (
                    self.features
                )

            # ---------------------------------------------------------------
            # ADD MISSING FEATURES
            # ---------------------------------------------------------------

            missing_features = []

            for feature in expected_features:

                if feature not in X.columns:

                    X[feature] = np.nan

                    missing_features.append(
                        feature
                    )

            # ---------------------------------------------------------------
            # SELECT FEATURES
            # ---------------------------------------------------------------

            X = X[
                expected_features
            ].copy()

            missing_ratio = (
                len(missing_features)
                /
                max(
                    len(expected_features),
                    1
                )
            )

            # ---------------------------------------------------------------
            # PREDICTION
            # ---------------------------------------------------------------

            prediction = (
                self.model.predict(X)[0]
            )

            # ---------------------------------------------------------------
            # PROBABILITY
            # ---------------------------------------------------------------

            probabilities = None

            if hasattr(
                self.model,
                "predict_proba"
            ):

                probabilities = (
                    self.model.predict_proba(X)[0]
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

            latency_ms = (
                time.perf_counter()
                -
                start_time
            ) * 1000.0

            return {

                "agent":
                    self.name,

                "task_type":
                    "fibrosis_classification",

                "model":
                    self.model_name,

                "status":
                    "success",

                "prediction":
                    str(prediction),

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
                        "fibrosis_classification",

                    "disease":
                        "fibrosis",

                    "features":
                        expected_features,

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
                    "fibrosis_classification",

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
                        "fibrosis_classification",

                    "disease":
                        "fibrosis"
                },

                "error":
                    str(e)
            }
