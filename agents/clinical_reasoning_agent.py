import numpy as np
import pandas as pd


class ClinicalReasoningAgent:

    def __init__(self, model_package):

        self.name = "ClinicalReasoningAgent"

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

        self.target_classes = model_package.get(
            "target_classes",
            ["0", "1"]
        )

    # ================================================================
    # CREATE DATAFRAME
    # ================================================================

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
                    columns=[
                        self.target_name
                    ]
                )

            # ========================================================
            # REMOVE UNKNOWN COLUMNS
            # ========================================================

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
            # FINAL ORDER
            # ========================================================

            X = X[
                self.feature_names
            ].copy()

            # ========================================================
            # NUMERICAL CONVERSION
            # ========================================================

            for col in self.feature_names:

                X[col] = pd.to_numeric(
                    X[col],
                    errors="coerce"
                )

            # ========================================================
            # MEDIAN IMPUTATION
            # ========================================================

            for col in self.feature_names:

                if X[col].isna().any():

                    # TabNet requires valid numeric values.
                    # Use a neutral fallback if missing.

                    X[col] = X[col].fillna(
                        X[col].median()
                    )

                    if X[col].isna().all():

                        X[col] = X[col].fillna(
                            0.0
                        )

            # ========================================================
            # NUMPY
            # ========================================================

            X_np = X.values.astype(
                np.float32
            )

            # ========================================================
            # PREDICTION
            # ========================================================

            prediction_encoded = int(
                self.model.predict(
                    X_np
                )[0]
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
                    self.model
                    .predict_proba(X_np)[0]
                )

            # ========================================================
            # DECODE
            # ========================================================

            if (
                prediction_encoded
                < len(self.target_classes)
            ):

                prediction = str(
                    self.target_classes[
                        prediction_encoded
                    ]
                )

            else:

                prediction = str(
                    prediction_encoded
                )

            # ========================================================
            # CONFIDENCE
            # ========================================================

            if probabilities is not None:

                confidence = float(
                    np.max(
                        probabilities
                    )
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

            missing_count = int(
                X.isna().sum().sum()
            )

            if missing_count == 0:

                quality = 1.0

            else:

                quality = max(
                    0.0,
                    1.0
                    - (
                        missing_count
                        /
                        len(self.feature_names)
                    )
                )

            # ========================================================
            # CLINICAL INTERPRETATION
            # ========================================================

            if prediction == "1":

                interpretation = (
                    "Clinical profile classified "
                    "in class 1."
                )

                recommendation = (
                    "Consider additional clinical "
                    "assessment according to the "
                    "available patient information."
                )

            else:

                interpretation = (
                    "Clinical profile classified "
                    "in class 2."
                )

                recommendation = (
                    "Consider additional clinical "
                    "assessment and correlation "
                    "with other liver findings."
                )

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
                    float(quality),

                "class_probabilities":
                    probability_list,

                "details": {

                    "task_type":
                        "clinical_classification",

                    "disease":
                        "liver_disorder",

                    "target":
                        self.target_name,

                    "classes":
                        [
                            str(x)
                            for x in self.target_classes
                        ],

                    "features":
                        self.feature_names,

                    "interpretation":
                        interpretation,

                    "recommendation":
                        recommendation
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

                "details": {

                    "task_type":
                        "clinical_classification",

                    "disease":
                        "liver_disorder"
                },

                "status":
                    "error",

                "error":
                    str(e)
            }
