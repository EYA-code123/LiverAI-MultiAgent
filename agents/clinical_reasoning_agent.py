import numpy as np
import pandas as pd


class ClinicalReasoningAgent:

    def __init__(self, model_package):

        self.name = "ClinicalReasoningAgent"

        # ============================================================
        # VALIDATION DU MODEL PACKAGE
        # ============================================================

        if model_package is None:
            raise ValueError(
                "ClinicalReasoningAgent: model_package cannot be None."
            )

        if not isinstance(model_package, dict):
            raise TypeError(
                "ClinicalReasoningAgent: model_package must be a dictionary."
            )

        if "model" not in model_package:
            raise KeyError(
                "ClinicalReasoningAgent: 'model' is missing "
                "from model_package."
            )

        # ============================================================
        # MODEL
        # ============================================================

        self.model_name = model_package.get(
            "model_name",
            "TabNet"
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
                self.feature_names
            )
        )

        self.categorical_columns = list(
            model_package.get(
                "categorical_columns",
                []
            )
        )

        # ============================================================
        # TARGET
        # ============================================================

        self.target_name = model_package.get(
            "target_name",
            "selector"
        )

        self.target_classes = list(
            model_package.get(
                "target_classes",
                ["0", "1"]
            )
        )

    # ================================================================
    # CREATE DATAFRAME
    # ================================================================

    def _create_dataframe(self, patient_data):

        if isinstance(patient_data, pd.DataFrame):

            return patient_data.copy()

        if isinstance(patient_data, dict):

            return pd.DataFrame(
                [patient_data]
            )

        if patient_data is None:

            return pd.DataFrame(
                columns=self.feature_names
            )

        return pd.DataFrame(
            [patient_data],
            columns=self.feature_names
        )

    # ================================================================
    # PREPARE INPUT
    # ================================================================

    def _prepare_input(self, patient_data):

        X = self._create_dataframe(
            patient_data
        )

        # ============================================================
        # REMOVE TARGET
        # ============================================================

        if self.target_name in X.columns:

            X = X.drop(
                columns=[self.target_name]
            )

        # ============================================================
        # REMOVE UNKNOWN COLUMNS
        # ============================================================

        unknown_columns = [
            col
            for col in X.columns
            if col not in self.feature_names
        ]

        if unknown_columns:

            X = X.drop(
                columns=unknown_columns
            )

        # ============================================================
        # ADD MISSING FEATURES
        # ============================================================

        for col in self.feature_names:

            if col not in X.columns:

                X[col] = np.nan

        # ============================================================
        # FINAL FEATURE ORDER
        # ============================================================

        if self.feature_names:

            X = X[
                self.feature_names
            ].copy()

        # ============================================================
        # NUMERICAL CONVERSION
        # ============================================================

        for col in X.columns:

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            )

        # ============================================================
        # MISSING VALUES
        # ============================================================

        missing_before = int(
            X.isna().sum().sum()
        )

        # ============================================================
        # MEDIAN IMPUTATION
        # ============================================================

        for col in X.columns:

            if X[col].isna().any():

                median_value = X[col].median()

                if pd.isna(median_value):

                    median_value = 0.0

                X[col] = X[col].fillna(
                    median_value
                )

        # ============================================================
        # FINAL SAFETY CHECK
        # ============================================================

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        X = X.fillna(0.0)

        # ============================================================
        # QUALITY
        # ============================================================

        total_features = max(
            len(X.columns),
            1
        )

        quality = max(
            0.0,
            1.0
            - (
                missing_before
                /
                total_features
            )
        )

        return X, missing_before, quality

    # ================================================================
    # PREDICT
    # ================================================================

    def predict(self, patient_data):

        try:

            # ========================================================
            # PREPARE DATA
            # ========================================================

            X, missing_count, quality = (
                self._prepare_input(
                    patient_data
                )
            )

            # ========================================================
            # NUMPY
            # ========================================================

            X_np = X.values.astype(
                np.float32
            )

            # ========================================================
            # MODEL PREDICTION
            # ========================================================

            raw_prediction = self.model.predict(
                X_np
            )

            # ========================================================
            # HANDLE PREDICTION SHAPE
            # ========================================================

            raw_prediction = np.asarray(
                raw_prediction
            )

            if raw_prediction.ndim > 1:

                raw_prediction = (
                    raw_prediction.reshape(-1)
                )

            prediction_raw = raw_prediction[0]

            # ========================================================
            # CONVERT PREDICTION
            # ========================================================

            try:

                prediction_encoded = int(
                    prediction_raw
                )

            except Exception:

                prediction_encoded = prediction_raw

            # ========================================================
            # PROBABILITIES
            # ========================================================

            probabilities = None

            if hasattr(
                self.model,
                "predict_proba"
            ):

                try:

                    probabilities = (
                        self.model
                        .predict_proba(X_np)
                    )

                    probabilities = np.asarray(
                        probabilities
                    )

                    if probabilities.ndim > 1:

                        probabilities = (
                            probabilities[0]
                        )

                    else:

                        probabilities = (
                            probabilities.reshape(-1)
                        )

                except Exception:

                    probabilities = None

            # ========================================================
            # DECODE CLASS
            # ========================================================

            if isinstance(
                prediction_encoded,
                (int, np.integer)
            ):

                prediction_index = int(
                    prediction_encoded
                )

                if (
                    0 <= prediction_index
                    < len(self.target_classes)
                ):

                    prediction = str(
                        self.target_classes[
                            prediction_index
                        ]
                    )

                else:

                    prediction = str(
                        prediction_encoded
                    )

            else:

                prediction = str(
                    prediction_encoded
                )

            # ========================================================
            # CONFIDENCE
            # ========================================================

            confidence = None
            probability_list = None

            if probabilities is not None:

                probability_list = [
                    float(x)
                    for x in probabilities
                ]

                if len(probability_list) > 0:

                    confidence = float(
                        np.max(
                            probabilities
                        )
                    )

            # ========================================================
            # UNCERTAINTY
            # ========================================================

            uncertainty = None

            if confidence is not None:

                uncertainty = float(
                    1.0 - confidence
                )

            # ========================================================
            # CLINICAL INTERPRETATION
            # ========================================================

            if prediction == "1":

                interpretation = (
                    "Clinical profile classified "
                    "in class 1."
                )

            elif prediction == "0":

                interpretation = (
                    "Clinical profile classified "
                    "in class 0."
                )

            else:

                interpretation = (
                    f"Clinical profile classified "
                    f"in class {prediction}."
                )

            # ========================================================
            # RESULT
            # ========================================================

            result = {

                "agent": self.name,

                "model_name": self.model_name,

                "prediction": prediction,

                "prediction_encoded": (
                    int(prediction_encoded)
                    if isinstance(
                        prediction_encoded,
                        (int, np.integer)
                    )
                    else str(
                        prediction_encoded
                    )
                ),

                "confidence": confidence,

                "uncertainty": uncertainty,

                "probabilities": probability_list,

                "target_name": self.target_name,

                "target_classes": [
                    str(x)
                    for x in self.target_classes
                ],

                "quality": float(
                    quality
                ),

                "missing_features": int(
                    missing_count
                ),

                "interpretation": interpretation,

                "status": "success"
            }

            return result

        # ============================================================
        # ERROR HANDLING
        # ============================================================

        except Exception as e:

            return {

                "agent": self.name,

                "model_name": self.model_name,

                "prediction": None,

                "prediction_encoded": None,

                "confidence": None,

                "uncertainty": None,

                "probabilities": None,

                "target_name": self.target_name,

                "target_classes": [
                    str(x)
                    for x in self.target_classes
                ],

                "quality": 0.0,

                "missing_features": None,

                "interpretation": (
                    "Clinical reasoning prediction "
                    "could not be completed."
                ),

                "status": "error",

                "error": str(e)
            }
