# =============================================================================
# Cirrhosis Agent
# =============================================================================

import time
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

        # =========================================================================
        # MODEL FEATURES
        # =========================================================================

        if hasattr(self.model, "feature_names_in_"):

            self.feature_names = [
                str(x)
                for x in self.model.feature_names_in_
            ]

        else:

            self.feature_names = [
                str(x)
                for x in model_package.get(
                    "feature_names",
                    []
                )
            ]

        # =========================================================================
        # PREPROCESSING
        # =========================================================================

        self.numerical_columns = [
            str(x)
            for x in model_package.get(
                "numerical_columns",
                []
            )
        ]

        self.categorical_columns = [
            str(x)
            for x in model_package.get(
                "categorical_columns",
                []
            )
        ]

        self.encoders = model_package.get(
            "encoders",
            {}
        )

        self.target_encoder = model_package.get(
            "target_encoder"
        )

        self.numerical_imputer = model_package.get(
            "numerical_imputer"
        )

        self.categorical_imputer = model_package.get(
            "categorical_imputer"
        )

        # =========================================================================
        # TARGET
        # =========================================================================

        self.target_name = "Stage"

        self.classes = [
            "1.0",
            "2.0",
            "3.0"
        ]

        # =========================================================================
        # SAFETY CHECK
        # =========================================================================

        if self.target_name in self.feature_names:

            raise ValueError(
                "Invalid model package: "
                "Stage is present in XGBoost model features."
            )

    # =========================================================================
    # CREATE DATAFRAME
    # =========================================================================

    def _create_dataframe(self, patient_data):

        if isinstance(patient_data, dict):

            return pd.DataFrame(
                [patient_data]
            )

        if isinstance(patient_data, pd.DataFrame):

            return patient_data.copy()

        return pd.DataFrame(
            [patient_data],
            columns=self.feature_names
        )

    # =========================================================================
    # PREPROCESS
    # =========================================================================

    def _preprocess(self, patient_data):

        X = self._create_dataframe(
            patient_data
        )

        # -------------------------------------------------------------------------
        # Remove target if supplied by mistake
        # -------------------------------------------------------------------------

        if self.target_name in X.columns:

            X = X.drop(
                columns=[self.target_name]
            )

        # -------------------------------------------------------------------------
        # Model features
        # -------------------------------------------------------------------------

        model_features = [
            feature
            for feature in self.feature_names
            if feature != self.target_name
        ]

        # -------------------------------------------------------------------------
        # Add missing model features
        # -------------------------------------------------------------------------

        for feature in model_features:

            if feature not in X.columns:

                X[feature] = np.nan

        # =========================================================================
        # NUMERICAL IMPUTATION
        # =========================================================================

        numerical_features = [
            col
            for col in self.numerical_columns
            if col in model_features
        ]

        if (
            self.numerical_imputer is not None
            and numerical_features
        ):

            # The saved imputer was fitted with Stage.
            # Therefore we must reproduce its exact input schema.

            if hasattr(
                self.numerical_imputer,
                "feature_names_in_"
            ):

                imputer_features = [
                    str(x)
                    for x in self.numerical_imputer.feature_names_in_
                ]

            else:

                imputer_features = numerical_features

            X_num = pd.DataFrame(
                index=X.index
            )

            for feature in imputer_features:

                if feature == self.target_name:

                    # Stage is the target.
                    # Never use the patient's target here.
                    X_num[feature] = np.nan

                elif feature in X.columns:

                    X_num[feature] = X[feature]

                else:

                    X_num[feature] = np.nan

            # Exact order expected by the imputer
            X_num = X_num[
                imputer_features
            ]

            transformed = (
                self.numerical_imputer
                .transform(X_num)
            )

            transformed = pd.DataFrame(
                transformed,
                columns=imputer_features,
                index=X.index
            )

            # Copy only the numerical model features.
            for feature in numerical_features:

                if feature in transformed.columns:

                    X[feature] = transformed[feature]

        # =========================================================================
        # CATEGORICAL IMPUTATION
        # =========================================================================

        categorical_features = [
            col
            for col in self.categorical_columns
            if col in model_features
        ]

        if (
            self.categorical_imputer is not None
            and categorical_features
        ):

            if hasattr(
                self.categorical_imputer,
                "feature_names_in_"
            ):

                imputer_features = [
                    str(x)
                    for x in self.categorical_imputer.feature_names_in_
                ]

            else:

                imputer_features = categorical_features

            X_cat = pd.DataFrame(
                index=X.index
            )

            for feature in imputer_features:

                if feature in X.columns:

                    X_cat[feature] = X[feature]

                else:

                    X_cat[feature] = np.nan

            # Exact order expected by categorical imputer
            X_cat = X_cat[
                imputer_features
            ]

            transformed = (
                self.categorical_imputer
                .transform(X_cat)
            )

            transformed = pd.DataFrame(
                transformed,
                columns=imputer_features,
                index=X.index
            )

            for feature in categorical_features:

                if feature in transformed.columns:

                    X[feature] = transformed[feature]

        # =========================================================================
        # CATEGORICAL ENCODING
        # =========================================================================

        for feature in categorical_features:

            if feature not in self.encoders:

                continue

            encoder = self.encoders[
                feature
            ]

            values = X[
                feature
            ].astype(str)

            known_values = set(
                str(value)
                for value in encoder.classes_
            )

            # Unknown categories are replaced by
            # the first category known by the encoder.
            values = values.apply(
                lambda value:
                value
                if value in known_values
                else str(encoder.classes_[0])
            )

            X[feature] = encoder.transform(
                values
            )

        # =========================================================================
        # FINAL MODEL INPUT
        # =========================================================================

        X = X[
            model_features
        ].copy()

        # -------------------------------------------------------------------------
        # Absolute safety check
        # -------------------------------------------------------------------------

        if self.target_name in X.columns:

            raise ValueError(
                "CRITICAL ERROR: Stage is still present "
                "in XGBoost input."
            )

        # -------------------------------------------------------------------------
        # Match exact XGBoost feature order
        # -------------------------------------------------------------------------

        if hasattr(
            self.model,
            "feature_names_in_"
        ):

            model_order = [
                str(x)
                for x in self.model.feature_names_in_
            ]

            X = X[
                model_order
            ].copy()

        # -------------------------------------------------------------------------
        # Final validation
        # -------------------------------------------------------------------------

        if hasattr(
            self.model,
            "n_features_in_"
        ):

            if (
                X.shape[1]
                != self.model.n_features_in_
            ):

                raise ValueError(
                    "Final feature count mismatch: "
                    f"XGBoost expects "
                    f"{self.model.n_features_in_}, "
                    f"received {X.shape[1]}."
                )

        return X

    # =========================================================================
    # PREDICT
    # =========================================================================

    def predict(self, patient_data):

        start_time = time.perf_counter()

        try:

            # ---------------------------------------------------------------------
            # Preprocessing
            # ---------------------------------------------------------------------

            X = self._preprocess(
                patient_data
            )

            # ---------------------------------------------------------------------
            # Missing data ratio
            # ---------------------------------------------------------------------

            missing_values = (
                X.isna()
                .sum()
                .sum()
            )

            missing_ratio = float(
                missing_values
                /
                max(
                    X.shape[1],
                    1
                )
            )

            # =========================================================================
            # PREDICTION
            # =========================================================================

            prediction_encoded = (
                self.model.predict(X)[0]
            )

            # =========================================================================
            # PROBABILITIES
            # =========================================================================

            probabilities = None

            if hasattr(
                self.model,
                "predict_proba"
            ):

                probabilities = (
                    self.model
                    .predict_proba(X)[0]
                )

            # =========================================================================
            # CONFIDENCE
            # =========================================================================

            if probabilities is not None:

                confidence = float(
                    np.max(
                        probabilities
                    )
                )

                class_probabilities = [
                    float(value)
                    for value in probabilities
                ]

            else:

                confidence = 0.0

                class_probabilities = None

            # =========================================================================
            # UNCERTAINTY
            # =========================================================================

            uncertainty = float(
                1.0 - confidence
            )

            # =========================================================================
            # DATA QUALITY
            # =========================================================================

            quality = float(
                max(
                    0.0,
                    min(
                        1.0,
                        1.0 - missing_ratio
                    )
                )
            )

            # =========================================================================
            # DECODE PREDICTION
            # =========================================================================

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

                    prediction = prediction_encoded

            prediction = str(
                prediction
            )

            # =========================================================================
            # LATENCY
            # =========================================================================

            latency_ms = (
                time.perf_counter()
                -
                start_time
            ) * 1000.0

            # =========================================================================
            # SUCCESS
            # =========================================================================

            return {

                "agent_id":
                    self.name,

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

                "explanation":
                    None,

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
                            if feature != self.target_name
                        ]
                },

                "error":
                    None
            }

        # =========================================================================
        # ERROR
        # =========================================================================

        except Exception as e:

            latency_ms = (
                time.perf_counter()
                -
                start_time
            ) * 1000.0

            return {

                "agent_id":
                    self.name,

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

                "explanation":
                    None,

                "error":
                    str(e)
            }
