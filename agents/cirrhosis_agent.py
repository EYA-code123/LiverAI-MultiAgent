import numpy as np
import pandas as pd


class CirrhosisAgent:

    def __init__(self, model_package):

        self.name = "CirrhosisAgent"

        self.model_name = model_package.get(
            "model_name",
            "XGBoost"
        )

        # ==========================================================
        # MODEL
        # ==========================================================

        self.model = model_package["model"]

        # ==========================================================
        # FEATURES
        # ==========================================================

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

        # ==========================================================
        # PREPROCESSING
        # ==========================================================

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
        # TARGET
        # ==========================================================

        self.target_name = "Stage"

        self.classes = [
            "1.0",
            "2.0",
            "3.0"
        ]

    # ==============================================================
    # CREATE DATAFRAME
    # ==============================================================

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

    # ==============================================================
    # PREPROCESS
    # ==============================================================

    def _preprocess(self, patient_data):

        X = self._create_dataframe(
            patient_data
        )

        # ----------------------------------------------------------
        # TARGET MUST NOT BE USED AS INPUT
        # ----------------------------------------------------------

        if self.target_name in X.columns:

            X = X.drop(
                columns=[self.target_name]
            )

        # ----------------------------------------------------------
        # MODEL FEATURE CHECK
        # ----------------------------------------------------------

        expected_features = [
            feature
            for feature in self.feature_names
            if feature != self.target_name
        ]

        # ----------------------------------------------------------
        # ADD MISSING FEATURES
        # ----------------------------------------------------------

        for col in expected_features:

            if col not in X.columns:

                X[col] = np.nan

        # ----------------------------------------------------------
        # NUMERICAL IMPUTATION
        # ----------------------------------------------------------

        numerical_features = [
            col
            for col in self.numerical_columns
            if col in X.columns
            and col != self.target_name
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

        # ----------------------------------------------------------
        # CATEGORICAL IMPUTATION
        # ----------------------------------------------------------

        categorical_features = [
            col
            for col in self.categorical_columns
            if col in X.columns
            and col != self.target_name
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

        # ----------------------------------------------------------
        # CATEGORICAL ENCODING
        # ----------------------------------------------------------

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

            # Unknown values are replaced by
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

        # ----------------------------------------------------------
        # FINAL FEATURE LIST
        # ----------------------------------------------------------

        X = X[
            expected_features
        ].copy()

        return X

    # ==============================================================
    # PREDICT
    # ==============================================================

    def predict(self, patient_data):

        try:

            # ------------------------------------------------------
            # PREPROCESS
            # ------------------------------------------------------

            X = self._preprocess(
                patient_data
            )

            # ------------------------------------------------------
            # IMPORTANT MODEL VALIDATION
            # ------------------------------------------------------

            model_feature_count = getattr(
                self.model,
                "n_features_in_",
                None
            )

            expected_feature_count = len(
                self.feature_names
            )

            # The current package contains Stage
            # in feature_names although Stage is the target.
            if (
                self.target_name in self.feature_names
            ):

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

                    "status":
                        "error",

                    "details": {

                        "task_type":
                            "classification",

                        "disease":
                            "cirrhosis",

                        "classes":
                            self.classes,

                        "problem":
                            "Target Stage is incorrectly "
                            "included in the saved model "
                            "feature list.",

                        "expected_model_features":
                            expected_feature_count,

                        "usable_input_features":
                            len(X.columns)
                    },

                    "error":
                        (
                            "The saved XGBoost package "
                            "contains the target 'Stage' "
                            "inside feature_names. "
                            "The model must be re-saved "
                            "without Stage as an input feature."
                        )
                }

            # ------------------------------------------------------
            # FEATURE COUNT
            # ------------------------------------------------------

            if (
                model_feature_count is not None
                and model_feature_count != len(X.columns)
            ):

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

                    "status":
                        "error",

                    "details": {

                        "task_type":
                            "classification",

                        "disease":
                            "cirrhosis",

                        "model_features":
                            model_feature_count,

                        "input_features":
                            len(X.columns)
                    },

                    "error":
                        (
                            "Feature count mismatch: "
                            f"model expects "
                            f"{model_feature_count} features, "
                            f"but the agent provides "
                            f"{len(X.columns)}."
                        )
                }

            # ------------------------------------------------------
            # PREDICTION
            # ------------------------------------------------------

            prediction_encoded = (
                self.model.predict(X)[0]
            )

            # ------------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------------

            probabilities = None

            if hasattr(
                self.model,
                "predict_proba"
            ):

                probabilities = (
                    self.model
                    .predict_proba(X)[0]
                )

            # ------------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------------

            if probabilities is not None:

                confidence = float(
                    np.max(
                        probabilities
                    )
                )

                uncertainty = float(
                    1.0 - confidence
                )

            else:

                confidence = None
                uncertainty = None

            # ------------------------------------------------------
            # DECODE TARGET
            # ------------------------------------------------------

            prediction = (
                prediction_encoded
            )

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

            # ------------------------------------------------------
            # QUALITY
            # ------------------------------------------------------

            if confidence is not None:

                quality = confidence

            else:

                quality = 0.0

            # ------------------------------------------------------
            # STANDARDIZED OUTPUT
            # ------------------------------------------------------

            result = {

                "agent":
                    self.name,

                "model":
                    self.model_name,

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

                "status":
                    "completed",

                "details": {

                    "task_type":
                        "classification",

                    "disease":
                        "cirrhosis",

                    "classes":
                        self.classes
                },

                "error":
                    None
            }

            # ------------------------------------------------------
            # CLASS PROBABILITIES
            # ------------------------------------------------------

            if probabilities is not None:

                result[
                    "class_probabilities"
                ] = [

                    float(value)
                    for value in probabilities

                ]

            return result

        # ==========================================================
        # GLOBAL ERROR HANDLING
        # ==========================================================

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

                "status":
                    "error",

                "details": {

                    "task_type":
                        "classification",

                    "disease":
                        "cirrhosis"
                },

                "error":
                    str(e)
            }
