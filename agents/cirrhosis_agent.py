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

        # ---------------------------------------------------------------------
        # MODEL FEATURES
        # ---------------------------------------------------------------------

        # The XGBoost model is the final authority for the input features.
        if hasattr(self.model, "feature_names_in_"):

            self.feature_names = list(
                self.model.feature_names_in_
            )

        else:

            self.feature_names = list(
                model_package.get(
                    "feature_names",
                    []
                )
            )

        # ---------------------------------------------------------------------
        # PREPROCESSING INFORMATION
        # ---------------------------------------------------------------------

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
            "target_encoder"
        )

        self.numerical_imputer = model_package.get(
            "numerical_imputer"
        )

        self.categorical_imputer = model_package.get(
            "categorical_imputer"
        )

        # ---------------------------------------------------------------------
        # TARGET
        # ---------------------------------------------------------------------

        self.target_name = "Stage"

        self.classes = [
            "1.0",
            "2.0",
            "3.0"
        ]

        # ---------------------------------------------------------------------
        # SAFETY CHECK
        # ---------------------------------------------------------------------

        if self.target_name in self.feature_names:

            raise ValueError(
                "Invalid model package: "
                "Stage is present in model features."
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

        # ---------------------------------------------------------------------
        # CREATE DATAFRAME
        # ---------------------------------------------------------------------

        X = self._create_dataframe(
            patient_data
        )

        # ---------------------------------------------------------------------
        # TARGET MUST NEVER BE USED
        # ---------------------------------------------------------------------

        if self.target_name in X.columns:

            X = X.drop(
                columns=[self.target_name]
            )

        # ---------------------------------------------------------------------
        # EXPECTED FEATURES
        # ---------------------------------------------------------------------

        expected_features = [
            feature
            for feature in self.feature_names
            if feature != self.target_name
        ]

        # ---------------------------------------------------------------------
        # ADD MISSING FEATURES
        # ---------------------------------------------------------------------

        for feature in expected_features:

            if feature not in X.columns:

                X[feature] = np.nan

        # ---------------------------------------------------------------------
        # KEEP ONLY EXPECTED FEATURES
        # ---------------------------------------------------------------------

        X = X[
            expected_features
        ].copy()

        # =====================================================================
        # NUMERICAL IMPUTATION
        # =====================================================================

        numerical_features = [
            col
            for col in self.numerical_columns
            if col in expected_features
        ]

        if (
            self.numerical_imputer is not None
            and numerical_features
        ):

            try:

                # Check what the imputer expects.
                if hasattr(
                    self.numerical_imputer,
                    "feature_names_in_"
                ):

                    imputer_features = list(
                        self.numerical_imputer.feature_names_in_
                    )

                else:

                    imputer_features = numerical_features

                # Build exactly what the imputer expects.
                X_num = pd.DataFrame(
                    index=X.index
                )

                for col in imputer_features:

                    if col in X.columns:

                        X_num[col] = X[col]

                    else:

                        X_num[col] = np.nan

                # If the imputer was incorrectly fitted with Stage,
                # provide an empty Stage column only to the imputer.
                if (
                    self.target_name
                    in imputer_features
                    and self.target_name not in X_num.columns
                ):

                    X_num[self.target_name] = np.nan

                # Exact column order.
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

                # Copy only real model features back.
                for col in numerical_features:

                    if col in transformed.columns:

                        X[col] = transformed[col]

            except Exception as e:

                raise ValueError(
                    "Numerical preprocessing error: "
                    + str(e)
                )

        # =====================================================================
        # CATEGORICAL IMPUTATION
        # =====================================================================

        categorical_features = [
            col
            for col in self.categorical_columns
            if col in expected_features
        ]

        if (
            self.categorical_imputer is not None
            and categorical_features
        ):

            try:

                # Check what the imputer expects.
                if hasattr(
                    self.categorical_imputer,
                    "feature_names_in_"
                ):

                    imputer_features = list(
                        self.categorical_imputer.feature_names_in_
                    )

                else:

                    imputer_features = categorical_features

                # Build exactly what the imputer expects.
                X_cat = pd.DataFrame(
                    index=X.index
                )

                for col in imputer_features:

                    if col in X.columns:

                        X_cat[col] = X[col]

                    else:

                        X_cat[col] = np.nan

                # If Stage was included when the imputer was fitted,
                # give it a temporary empty value.
                if (
                    self.target_name
                    in imputer_features
                    and self.target_name not in X_cat.columns
                ):

                    X_cat[self.target_name] = np.nan

                # Exact column order.
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

                # Copy only real categorical model features.
                for col in categorical_features:

                    if col in transformed.columns:

                        X[col] = transformed[col]

            except Exception as e:

                raise ValueError(
                    "Categorical preprocessing error: "
                    + str(e)
                )

        # =====================================================================
        # CATEGORICAL ENCODING
        # =====================================================================

        for col in categorical_features:

            if col not in self.encoders:

                continue

            encoder = self.encoders[col]

            values = X[col].astype(str)

            known_values = set(
                encoder.classes_
            )

            # Unknown categories are replaced by
            # the first known category.
            values = values.apply(
                lambda value:
                value
                if value in known_values
                else encoder.classes_[0]
            )

            X[col] = encoder.transform(
                values
            )

        # =====================================================================
        # FINAL FEATURE ORDER
        # =====================================================================

        X = X[
            expected_features
        ].copy()

        # ---------------------------------------------------------------------
        # FINAL SAFETY CHECK
        # ---------------------------------------------------------------------

        if self.target_name in X.columns:

            raise ValueError(
                "CRITICAL ERROR: Stage is present "
                "in the prediction input."
            )

        # ---------------------------------------------------------------------
        # MATCH XGBOOST FEATURE ORDER
        # ---------------------------------------------------------------------

        if hasattr(
            self.model,
            "feature_names_in_"
        ):

            model_features = list(
                self.model.feature_names_in_
            )

            X = X[
                model_features
            ].copy()

        # ---------------------------------------------------------------------
        # FINAL CHECK
        # ---------------------------------------------------------------------

        if list(X.columns) != expected_features:

            # If the model provides its own feature list,
            # use that list as the final reference.
            if hasattr(
                self.model,
                "feature_names_in_"
            ):

                model_features = list(
                    self.model.feature_names_in_
                )

                X = X[
                    model_features
                ].copy()

        return X

    # =========================================================================
    # PREDICT
    # =========================================================================

    def predict(self, patient_data):

        start_time = time.perf_counter()

        try:

            # -----------------------------------------------------------------
            # PREPROCESS
            # -----------------------------------------------------------------

            X = self._preprocess(
                patient_data
            )

            # -----------------------------------------------------------------
            # FEATURE VALIDATION
            # -----------------------------------------------------------------

            if self.target_name in X.columns:

                raise ValueError(
                    "Stage must not be present "
                    "in model input."
                )

            model_feature_count = getattr(
                self.model,
                "n_features_in_",
                None
            )

            if (
                model_feature_count is not None
                and model_feature_count
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

            # -----------------------------------------------------------------
            # MODEL PREDICTION
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

            # -----------------------------------------------------------------
            # CONFIDENCE
            # -----------------------------------------------------------------

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

            # -----------------------------------------------------------------
            # UNCERTAINTY
            # -----------------------------------------------------------------

            uncertainty = float(
                1.0 - confidence
            )

            # -----------------------------------------------------------------
            # QUALITY
            # -----------------------------------------------------------------

            quality = float(
                max(
                    0.0,
                    min(
                        1.0,
                        1.0 - missing_ratio
                    )
                )
            )

            # -----------------------------------------------------------------
            # DECODE PREDICTION
            # -----------------------------------------------------------------

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

            # -----------------------------------------------------------------
            # LATENCY
            # -----------------------------------------------------------------

            latency_ms = (
                time.perf_counter()
                -
                start_time
            ) * 1000.0

            # -----------------------------------------------------------------
            # SUCCESS RESULT
            # -----------------------------------------------------------------

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
