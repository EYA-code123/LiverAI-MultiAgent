# =============================================================================
# Cirrhosis Agent
# =============================================================================

import time
import numpy as np
import pandas as pd


class CirrhosisAgent:

    def __init__(
        self,
        model_package
    ):

        self.name = (
            "CirrhosisAgent"
        )

        self.model_name = (
            model_package.get(
                "model_name",
                "XGBoost"
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

        self.target_encoder = (
            model_package.get(
                "target_encoder"
            )
        )

        self.numerical_imputer = (
            model_package.get(
                "numerical_imputer"
            )
        )

        self.categorical_imputer = (
            model_package.get(
                "categorical_imputer"
            )
        )

        self.target_name = "Stage"

        self.classes = [
            "1.0",
            "2.0",
            "3.0"
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
            columns=self.feature_names
        )

    # =========================================================================
    # PREPROCESS
    # =========================================================================

    def _preprocess(
        self,
        patient_data
    ):

        X = self._create_dataframe(
            patient_data
        )

        # ---------------------------------------------------------------------
        # REMOVE TARGET
        # ---------------------------------------------------------------------

        if self.target_name in X.columns:

            X = X.drop(
                columns=[
                    self.target_name
                ]
            )

        expected_features = [

            feature

            for feature in
            self.feature_names

            if feature
            != self.target_name
        ]

        # ---------------------------------------------------------------------
        # ADD MISSING FEATURES
        # ---------------------------------------------------------------------

        for col in expected_features:

            if col not in X.columns:

                X[col] = np.nan

        # ---------------------------------------------------------------------
        # NUMERICAL IMPUTATION
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # CATEGORICAL IMPUTATION
        # ---------------------------------------------------------------------

        categorical_features = [

            col

            for col in
            self.categorical_columns

            if col in X.columns
        ]

        if (
            self.categorical_imputer
            is not None
            and categorical_features
        ):

            X[
                categorical_features
            ] = (
                self.categorical_imputer
                .transform(
                    X[
                        categorical_features
                    ]
                )
            )

        # ---------------------------------------------------------------------
        # ENCODING
        # ---------------------------------------------------------------------

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

        X = X[
            expected_features
        ].copy()

        return X

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

            X = self._preprocess(
                patient_data
            )

            # -----------------------------------------------------------------
            # MODEL FEATURE CHECK
            # -----------------------------------------------------------------

            model_feature_count = (
                getattr(
                    self.model,
                    "n_features_in_",
                    None
                )
            )

            if (
                model_feature_count
                is not None
                and
                model_feature_count
                != len(X.columns)
            ):

                raise ValueError(

                    "Feature count mismatch: "
                    f"model expects "
                    f"{model_feature_count}, "
                    f"but input contains "
                    f"{len(X.columns)}."
                )

            # -----------------------------------------------------------------
            # MISSING DATA
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
            # PREDICT
            # -----------------------------------------------------------------

            prediction_encoded = (
                self.model.predict(X)[0]
            )

            # -----------------------------------------------------------------
            # PROBABILITY
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

                    float(value)

                    for value
                    in probabilities
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

            prediction = (
                prediction_encoded
            )

            if (
                self.target_encoder
                is not None
            ):

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

            prediction = str(
                prediction
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
                    "cirrhosis_classification",

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
                        "cirrhosis_classification",

                    "disease":
                        "cirrhosis",

                    "classes":
                        self.classes,

                    "features":
                        [
                            feature

                            for feature
                            in self.feature_names

                            if feature
                            != self.target_name
                        ]
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
                    "cirrhosis_classification",

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
                        "cirrhosis_classification",

                    "disease":
                        "cirrhosis"
                },

                "error":
                    str(e)
            }
