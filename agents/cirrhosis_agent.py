# =============================================================================
# LiverAI-MultiAgent
# Cirrhosis Agent
# =============================================================================
#
# Responsible for:
#   - Loading a trained cirrhosis model/package
#   - Validating model compatibility
#   - Preparing patient clinical data
#   - Handling missing values
#   - Handling categorical features
#   - Preserving the exact model feature order
#   - Performing prediction
#   - Returning probabilities/confidence
#   - Returning standardized agent output
#
# IMPORTANT:
# This agent NEVER uses the target column as an input feature.
#
# =============================================================================

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class CirrhosisAgent:
    """
    Agent responsible for cirrhosis classification.

    Expected model package:

        {
            "model": trained_model,

            # Optional
            "model_name": "XGBoost",

            "feature_names": [...],

            "numerical_columns": [...],

            "categorical_columns": [...],

            "encoders": {...},

            "target_encoder": encoder,

            "numerical_imputer": imputer,

            "categorical_imputer": imputer
        }

    The agent also supports a direct sklearn/XGBoost model object
    when it exposes feature_names_in_.
    """

    # =========================================================================
    # CONSTANTS
    # =========================================================================

    AGENT_NAME = "CirrhosisAgent"

    TASK_TYPE = "cirrhosis_classification"

    DISEASE = "cirrhosis"

    # IMPORTANT:
    # The current cirrhosis dataset/project uses Cirrhosis_Status,
    # NOT Stage.
    TARGET_NAME = "Cirrhosis_Status"

    DEFAULT_MODEL_NAME = "XGBoost"

    # Known semantic classes for the current project.
    #
    # 0 = Compensated
    # 1 = Decompensated
    # 2 = No_Cirrhosis
    #
    # We do NOT blindly force these labels if the loaded model contains
    # its own classes_ metadata.
    DEFAULT_CLASSES = {
        0: "Compensated",
        1: "Decompensated",
        2: "No_Cirrhosis",
    }

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, model_package: Any):

        self.name = self.AGENT_NAME

        # ---------------------------------------------------------------------
        # Validate package
        # ---------------------------------------------------------------------

        if model_package is None:
            raise ValueError(
                "CirrhosisAgent received an empty model package."
            )

        # ---------------------------------------------------------------------
        # Support dictionary/package
        # ---------------------------------------------------------------------

        if isinstance(model_package, dict):

            if "model" not in model_package:
                raise ValueError(
                    "Invalid cirrhosis model package: "
                    "missing required key 'model'."
                )

            self.model_package = model_package

            self.model = model_package["model"]

            self.model_name = str(
                model_package.get(
                    "model_name",
                    self.DEFAULT_MODEL_NAME
                )
            )

        # ---------------------------------------------------------------------
        # Support direct model object
        # ---------------------------------------------------------------------

        else:

            self.model_package = {
                "model": model_package
            }

            self.model = model_package

            self.model_name = self.DEFAULT_MODEL_NAME

        # ---------------------------------------------------------------------
        # Validate model
        # ---------------------------------------------------------------------

        if self.model is None:
            raise ValueError(
                "Cirrhosis model is None."
            )

        if not hasattr(self.model, "predict"):
            raise TypeError(
                "Invalid cirrhosis model: "
                "the object does not provide a predict() method."
            )

        # =========================================================================
        # FEATURE INFORMATION
        # =========================================================================

        self.feature_names = self._discover_feature_names()

        self.numerical_columns = self._get_list_from_package(
            "numerical_columns"
        )

        self.categorical_columns = self._get_list_from_package(
            "categorical_columns"
        )

        self.encoders = self._get_dict_from_package(
            "encoders"
        )

        self.target_encoder = self.model_package.get(
            "target_encoder"
        )

        self.numerical_imputer = self.model_package.get(
            "numerical_imputer"
        )

        self.categorical_imputer = self.model_package.get(
            "categorical_imputer"
        )

        # =========================================================================
        # TARGET
        # =========================================================================

        self.target_name = self.TARGET_NAME

        self.classes = self._discover_classes()

        # =========================================================================
        # SAFETY VALIDATION
        # =========================================================================

        self._validate_model_configuration()

    # =========================================================================
    # PACKAGE HELPERS
    # =========================================================================

    def _get_list_from_package(
        self,
        key: str
    ) -> List[str]:

        value = self.model_package.get(key, [])

        if value is None:
            return []

        if not isinstance(value, (list, tuple, np.ndarray)):
            return []

        return [
            str(item)
            for item in value
        ]

    # -------------------------------------------------------------------------

    def _get_dict_from_package(
        self,
        key: str
    ) -> Dict[str, Any]:

        value = self.model_package.get(
            key,
            {}
        )

        if value is None:
            return {}

        if isinstance(value, dict):
            return value

        return {}

    # =========================================================================
    # FEATURE DISCOVERY
    # =========================================================================

    def _discover_feature_names(self) -> List[str]:

        # ---------------------------------------------------------------------
        # Priority 1:
        # Model feature_names_in_
        # ---------------------------------------------------------------------

        if hasattr(
            self.model,
            "feature_names_in_"
        ):

            names = getattr(
                self.model,
                "feature_names_in_"
            )

            if names is not None:

                names = [
                    str(x)
                    for x in names
                ]

                if names:
                    return names

        # ---------------------------------------------------------------------
        # Priority 2:
        # Model package feature_names
        # ---------------------------------------------------------------------

        package_features = self.model_package.get(
            "feature_names",
            []
        )

        if package_features:

            return [
                str(x)
                for x in package_features
            ]

        # ---------------------------------------------------------------------
        # Priority 3:
        # Infer from numerical + categorical columns
        # ---------------------------------------------------------------------

        inferred = []

        for feature in self.numerical_columns:

            if feature not in inferred:
                inferred.append(feature)

        for feature in self.categorical_columns:

            if feature not in inferred:
                inferred.append(feature)

        return inferred

    # =========================================================================
    # CLASS DISCOVERY
    # =========================================================================

    def _discover_classes(self) -> List[str]:

        # ---------------------------------------------------------------------
        # Target encoder
        # ---------------------------------------------------------------------

        if self.target_encoder is not None:

            if hasattr(
                self.target_encoder,
                "classes_"
            ):

                try:

                    return [
                        str(x)
                        for x in self.target_encoder.classes_
                    ]

                except Exception:
                    pass

        # ---------------------------------------------------------------------
        # Model classes_
        # ---------------------------------------------------------------------

        if hasattr(
            self.model,
            "classes_"
        ):

            try:

                raw_classes = list(
                    self.model.classes_
                )

                decoded = []

                for value in raw_classes:

                    if isinstance(
                        value,
                        (int, np.integer)
                    ):

                        decoded.append(
                            self.DEFAULT_CLASSES.get(
                                int(value),
                                str(value)
                            )
                        )

                    else:

                        decoded.append(
                            str(value)
                        )

                return decoded

            except Exception:
                pass

        # ---------------------------------------------------------------------
        # Default project classes
        # ---------------------------------------------------------------------

        return [
            self.DEFAULT_CLASSES[0],
            self.DEFAULT_CLASSES[1],
            self.DEFAULT_CLASSES[2]
        ]

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate_model_configuration(self) -> None:

        # ---------------------------------------------------------------------
        # Target must NEVER be a model feature
        # ---------------------------------------------------------------------

        if self.target_name in self.feature_names:

            raise ValueError(
                "CRITICAL MODEL CONFIGURATION ERROR: "
                f"target '{self.target_name}' is present "
                "inside model features."
            )

        # ---------------------------------------------------------------------
        # Stage must also never be used as a target/input accidentally
        # ---------------------------------------------------------------------

        if "Stage" in self.feature_names:

            raise ValueError(
                "CRITICAL MODEL CONFIGURATION ERROR: "
                "'Stage' is present in cirrhosis model features. "
                "Verify that the correct trained model is being loaded."
            )

        # ---------------------------------------------------------------------
        # Validate feature count if available
        # ---------------------------------------------------------------------

        if hasattr(
            self.model,
            "n_features_in_"
        ):

            expected = int(
                self.model.n_features_in_
            )

            if self.feature_names:

                actual = len(
                    self.feature_names
                )

                if actual != expected:

                    raise ValueError(
                        "Cirrhosis model feature mismatch: "
                        f"model expects {expected} features, "
                        f"but metadata contains {actual}."
                    )

    # =========================================================================
    # DATAFRAME CREATION
    # =========================================================================

    def _create_dataframe(
        self,
        patient_data: Any
    ) -> pd.DataFrame:

        # ---------------------------------------------------------------------
        # Dictionary
        # ---------------------------------------------------------------------

        if isinstance(
            patient_data,
            dict
        ):

            return pd.DataFrame(
                [patient_data]
            )

        # ---------------------------------------------------------------------
        # DataFrame
        # ---------------------------------------------------------------------

        if isinstance(
            patient_data,
            pd.DataFrame
        ):

            return patient_data.copy()

        # ---------------------------------------------------------------------
        # Series
        # ---------------------------------------------------------------------

        if isinstance(
            patient_data,
            pd.Series
        ):

            return pd.DataFrame(
                [patient_data.to_dict()]
            )

        # ---------------------------------------------------------------------
        # Array/list
        # ---------------------------------------------------------------------

        if isinstance(
            patient_data,
            (list, tuple, np.ndarray)
        ):

            if not self.feature_names:

                raise ValueError(
                    "Cannot interpret array input because "
                    "model feature names are unavailable."
                )

            return pd.DataFrame(
                [patient_data],
                columns=self.feature_names
            )

        # ---------------------------------------------------------------------
        # Unsupported input
        # ---------------------------------------------------------------------

        raise TypeError(
            "Unsupported patient_data type: "
            f"{type(patient_data).__name__}. "
            "Expected dict, DataFrame, Series, list, tuple, or ndarray."
        )

    # =========================================================================
    # COLUMN NORMALIZATION
    # =========================================================================

    def _normalize_columns(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:

        X = X.copy()

        X.columns = [
            str(column)
            for column in X.columns
        ]

        return X

    # =========================================================================
    # PREPROCESSING
    # =========================================================================

    def _preprocess(
        self,
        patient_data: Any
    ) -> pd.DataFrame:

        X = self._create_dataframe(
            patient_data
        )

        X = self._normalize_columns(
            X
        )

        # ---------------------------------------------------------------------
        # Remove target if accidentally provided
        # ---------------------------------------------------------------------

        if self.target_name in X.columns:

            X = X.drop(
                columns=[
                    self.target_name
                ]
            )

        # ---------------------------------------------------------------------
        # Remove legacy target
        # ---------------------------------------------------------------------

        if "Stage" in X.columns:

            X = X.drop(
                columns=[
                    "Stage"
                ]
            )

        # =========================================================================
        # MODEL FEATURES
        # =========================================================================

        model_features = [
            feature
            for feature in self.feature_names
            if feature != self.target_name
            and feature != "Stage"
        ]

        if not model_features:

            raise ValueError(
                "No model features are available for cirrhosis prediction."
            )

        # =========================================================================
        # REMOVE EXTRA USER COLUMNS
        # =========================================================================

        # We intentionally keep only the features expected by the model.
        #
        # This protects against:
        #   - Patient_ID
        #   - target columns
        #   - unrelated clinical variables
        #   - accidental metadata

        X = X[
            [
                column
                for column in X.columns
                if column in model_features
            ]
        ].copy()

        # =========================================================================
        # ADD MISSING FEATURES
        # =========================================================================

        for feature in model_features:

            if feature not in X.columns:

                X[feature] = np.nan

        # =========================================================================
        # NUMERICAL FEATURES
        # =========================================================================

        numerical_features = [
            feature
            for feature in self.numerical_columns
            if feature in model_features
        ]

        # If numerical metadata is unavailable,
        # infer numeric columns from the model features.
        if not numerical_features:

            for feature in model_features:

                if feature in X.columns:

                    if pd.api.types.is_numeric_dtype(
                        X[feature]
                    ):

                        numerical_features.append(
                            feature
                        )

        # ---------------------------------------------------------------------
        # Convert numerical fields safely
        # ---------------------------------------------------------------------

        for feature in numerical_features:

            if feature in X.columns:

                X[feature] = pd.to_numeric(
                    X[feature],
                    errors="coerce"
                )

        # =========================================================================
        # NUMERICAL IMPUTATION
        # =========================================================================

        if (
            self.numerical_imputer is not None
            and numerical_features
        ):

            X = self._apply_numerical_imputer(
                X,
                numerical_features
            )

        # =========================================================================
        # CATEGORICAL FEATURES
        # =========================================================================

        categorical_features = [
            feature
            for feature in self.categorical_columns
            if feature in model_features
        ]

        # =========================================================================
        # CATEGORICAL IMPUTATION
        # =========================================================================

        if (
            self.categorical_imputer is not None
            and categorical_features
        ):

            X = self._apply_categorical_imputer(
                X,
                categorical_features
            )

        # =========================================================================
        # CATEGORICAL ENCODING
        # =========================================================================

        X = self._apply_categorical_encoders(
            X,
            categorical_features
        )

        # =========================================================================
        # FINAL COLUMN ORDER
        # =========================================================================

        X = X[
            model_features
        ].copy()

        # =========================================================================
        # MODEL FEATURE ORDER
        # =========================================================================

        if hasattr(
            self.model,
            "feature_names_in_"
        ):

            model_order = [
                str(x)
                for x in self.model.feature_names_in_
            ]

            # Safety: remove target if present
            model_order = [
                feature
                for feature in model_order
                if feature != self.target_name
                and feature != "Stage"
            ]

            missing = [
                feature
                for feature in model_order
                if feature not in X.columns
            ]

            if missing:

                raise ValueError(
                    "Missing required model features: "
                    + ", ".join(missing)
                )

            X = X[
                model_order
            ].copy()

        # =========================================================================
        # FINAL FEATURE COUNT CHECK
        # =========================================================================

        if hasattr(
            self.model,
            "n_features_in_"
        ):

            expected = int(
                self.model.n_features_in_
            )

            received = int(
                X.shape[1]
            )

            if received != expected:

                raise ValueError(
                    "Cirrhosis model input dimension mismatch: "
                    f"expected {expected} features, "
                    f"received {received}."
                )

        # =========================================================================
        # TARGET SAFETY CHECK
        # =========================================================================

        if self.target_name in X.columns:

            raise ValueError(
                "CRITICAL DATA LEAKAGE ERROR: "
                f"{self.target_name} is present in model input."
            )

        if "Stage" in X.columns:

            raise ValueError(
                "CRITICAL DATA LEAKAGE ERROR: "
                "Stage is present in model input."
            )

        return X

    # =========================================================================
    # NUMERICAL IMPUTER
    # =========================================================================

    def _apply_numerical_imputer(
        self,
        X: pd.DataFrame,
        numerical_features: List[str]
    ) -> pd.DataFrame:

        imputer = self.numerical_imputer

        if hasattr(
            imputer,
            "feature_names_in_"
        ):

            imputer_features = [
                str(x)
                for x in imputer.feature_names_in_
            ]

        else:

            imputer_features = numerical_features

        X_num = pd.DataFrame(
            index=X.index
        )

        for feature in imputer_features:

            if feature in X.columns:

                X_num[feature] = X[feature]

            else:

                X_num[feature] = np.nan

        X_num = X_num[
            imputer_features
        ]

        transformed = imputer.transform(
            X_num
        )

        transformed = pd.DataFrame(
            transformed,
            columns=imputer_features,
            index=X.index
        )

        for feature in numerical_features:

            if feature in transformed.columns:

                X[feature] = transformed[
                    feature
                ]

        return X

    # =========================================================================
    # CATEGORICAL IMPUTER
    # =========================================================================

    def _apply_categorical_imputer(
        self,
        X: pd.DataFrame,
        categorical_features: List[str]
    ) -> pd.DataFrame:

        imputer = self.categorical_imputer

        if hasattr(
            imputer,
            "feature_names_in_"
        ):

            imputer_features = [
                str(x)
                for x in imputer.feature_names_in_
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

        X_cat = X_cat[
            imputer_features
        ]

        transformed = imputer.transform(
            X_cat
        )

        transformed = pd.DataFrame(
            transformed,
            columns=imputer_features,
            index=X.index
        )

        for feature in categorical_features:

            if feature in transformed.columns:

                X[feature] = transformed[
                    feature
                ]

        return X

    # =========================================================================
    # CATEGORICAL ENCODERS
    # =========================================================================

    def _apply_categorical_encoders(
        self,
        X: pd.DataFrame,
        categorical_features: List[str]
    ) -> pd.DataFrame:

        for feature in categorical_features:

            if feature not in self.encoders:

                continue

            encoder = self.encoders[
                feature
            ]

            if not hasattr(
                encoder,
                "transform"
            ):

                raise TypeError(
                    f"Encoder for '{feature}' "
                    "does not provide transform()."
                )

            # -----------------------------------------------------------------
            # Convert values to strings only when the encoder uses strings.
            # -----------------------------------------------------------------

            values = X[
                feature
            ]

            # -----------------------------------------------------------------
            # Handle unknown categories safely
            # -----------------------------------------------------------------

            if hasattr(
                encoder,
                "classes_"
            ):

                known_values = set(
                    encoder.classes_
                )

                # Try exact values first.
                processed = []

                for value in values:

                    if pd.isna(value):

                        processed.append(
                            encoder.classes_[0]
                        )

                    elif value in known_values:

                        processed.append(
                            value
                        )

                    elif str(value) in {
                        str(x)
                        for x in encoder.classes_
                    }:

                        matching = next(
                            x
                            for x in encoder.classes_
                            if str(x) == str(value)
                        )

                        processed.append(
                            matching
                        )

                    else:

                        # Conservative fallback:
                        # use the first known category.
                        processed.append(
                            encoder.classes_[0]
                        )

                values = pd.Series(
                    processed,
                    index=X.index
                )

            transformed = encoder.transform(
                values
            )

            X[feature] = transformed

        return X

    # =========================================================================
    # PREDICTION DECODING
    # =========================================================================

    def _decode_prediction(
        self,
        prediction_encoded: Any
    ) -> str:

        # ---------------------------------------------------------------------
        # Target encoder
        # ---------------------------------------------------------------------

        if self.target_encoder is not None:

            if hasattr(
                self.target_encoder,
                "inverse_transform"
            ):

                try:

                    decoded = (
                        self.target_encoder
                        .inverse_transform(
                            [prediction_encoded]
                        )[0]
                    )

                    return str(
                        decoded
                    )

                except Exception:
                    pass

        # ---------------------------------------------------------------------
        # Numeric class mapping
        # ---------------------------------------------------------------------

        try:

            numeric_value = int(
                prediction_encoded
            )

            if numeric_value in self.DEFAULT_CLASSES:

                return self.DEFAULT_CLASSES[
                    numeric_value
                ]

        except Exception:
            pass

        return str(
            prediction_encoded
        )

    # =========================================================================
    # CLASS PROBABILITIES
    # =========================================================================

    def _build_class_probabilities(
        self,
        probabilities: Optional[np.ndarray]
    ) -> Optional[Dict[str, float]]:

        if probabilities is None:
            return None

        probabilities = np.asarray(
            probabilities
        ).reshape(-1)

        result = {}

        # ---------------------------------------------------------------------
        # Prefer model classes_
        # ---------------------------------------------------------------------

        if hasattr(
            self.model,
            "classes_"
        ):

            raw_classes = list(
                self.model.classes_
            )

        else:

            raw_classes = list(
                range(
                    len(probabilities)
                )
            )

        for index, probability in enumerate(
            probabilities
        ):

            if index >= len(raw_classes):
                break

            raw_class = raw_classes[
                index
            ]

            # Decode numeric project labels
            try:

                numeric_class = int(
                    raw_class
                )

                label = self.DEFAULT_CLASSES.get(
                    numeric_class,
                    str(raw_class)
                )

            except Exception:

                label = str(
                    raw_class
                )

            result[label] = float(
                probability
            )

        return result

    # =========================================================================
    # PREDICT
    # =========================================================================

    def predict(
        self,
        patient_data: Any
    ) -> Dict[str, Any]:

        start_time = time.perf_counter()

        try:

            # =================================================================
            # PREPROCESS
            # =================================================================

            X = self._preprocess(
                patient_data
            )

            # =================================================================
            # DATA QUALITY
            # =================================================================

            total_values = int(
                X.shape[0] * X.shape[1]
            )

            missing_values = int(
                X.isna()
                .sum()
                .sum()
            )

            if total_values > 0:

                missing_ratio = (
                    missing_values
                    /
                    total_values
                )

            else:

                missing_ratio = 1.0

            # =================================================================
            # PREDICTION
            # =================================================================

            raw_prediction = self.model.predict(
                X
            )

            if len(raw_prediction) == 0:

                raise RuntimeError(
                    "Cirrhosis model returned "
                    "an empty prediction."
                )

            prediction_encoded = (
                raw_prediction[0]
            )

            # =================================================================
            # PROBABILITIES
            # =================================================================

            probabilities_array = None

            if hasattr(
                self.model,
                "predict_proba"
            ):

                try:

                    probabilities_array = (
                        self.model
                        .predict_proba(X)[0]
                    )

                except Exception:
                    probabilities_array = None

            # =================================================================
            # CONFIDENCE
            # =================================================================

            if probabilities_array is not None:

                probabilities_array = np.asarray(
                    probabilities_array,
                    dtype=float
                )

                confidence = float(
                    np.max(
                        probabilities_array
                    )
                )

            else:

                confidence = 0.0

            confidence = max(
                0.0,
                min(
                    1.0,
                    confidence
                )
            )

            # =================================================================
            # UNCERTAINTY
            # =================================================================

            uncertainty = float(
                1.0 - confidence
            )

            # =================================================================
            # DATA QUALITY
            # =================================================================

            quality = float(
                1.0 - missing_ratio
            )

            quality = max(
                0.0,
                min(
                    1.0,
                    quality
                )
            )

            # =================================================================
            # DECODE PREDICTION
            # =================================================================

            prediction = self._decode_prediction(
                prediction_encoded
            )

            # =================================================================
            # CLASS PROBABILITIES
            # =================================================================

            class_probabilities = (
                self._build_class_probabilities(
                    probabilities_array
                )
            )

            # =================================================================
            # LATENCY
            # =================================================================

            latency_ms = (
                time.perf_counter()
                -
                start_time
            ) * 1000.0

            # =================================================================
            # RETURN
            # =================================================================

            return {

                "agent_id": self.name,

                "agent": self.name,

                "task_type": self.TASK_TYPE,

                "model": self.model_name,

                "status": "success",

                "prediction": prediction,

                "probability": confidence,

                "confidence": confidence,

                "uncertainty": uncertainty,

                "quality": quality,

                "missing_data_ratio": float(
                    missing_ratio
                ),

                "latency_ms": float(
                    latency_ms
                ),

                "class_probabilities":
                    class_probabilities,

                "explanation": None,

                "details": {

                    "task_type":
                        self.TASK_TYPE,

                    "disease":
                        self.DISEASE,

                    "target":
                        self.target_name,

                    "classes":
                        self.classes,

                    "features":
                        [
                            feature
                            for feature
                            in self.feature_names
                            if feature != self.target_name
                            and feature != "Stage"
                        ],

                    "n_features":
                        int(
                            X.shape[1]
                        ),

                    "missing_values":
                        missing_values
                },

                "error": None
            }

        # =====================================================================
        # ERROR HANDLING
        # =====================================================================

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
                    self.TASK_TYPE,

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
                    float(
                        latency_ms
                    ),

                "class_probabilities":
                    None,

                "details": {

                    "task_type":
                        self.TASK_TYPE,

                    "disease":
                        self.DISEASE,

                    "target":
                        self.target_name
                },

                "explanation":
                    None,

                "error":
                    str(e)
            }

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:

        """
        Check whether the agent/model is correctly initialized.
        """

        checks = {

            "agent_initialized":
                self is not None,

            "model_loaded":
                self.model is not None,

            "predict_available":
                hasattr(
                    self.model,
                    "predict"
                ),

            "features_available":
                len(
                    self.feature_names
                ) > 0,

            "target_not_in_features":
                self.target_name
                not in self.feature_names,

            "legacy_stage_not_in_features":
                "Stage"
                not in self.feature_names
        }

        healthy = all(
            checks.values()
        )

        return {

            "agent":
                self.name,

            "status":
                "healthy"
                if healthy
                else "unhealthy",

            "healthy":
                healthy,

            "checks":
                checks,

            "model":
                self.model_name,

            "target":
                self.target_name,

            "n_features":
                len(
                    self.feature_names
                ),

            "features":
                self.feature_names
        }

    # =========================================================================
    # REPRESENTATION
    # =========================================================================

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name!r}, "
            f"target={self.target_name!r}, "
            f"n_features={len(self.feature_names)}"
            ")"
        )
