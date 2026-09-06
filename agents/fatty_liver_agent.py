# =============================================================================
# FATTY LIVER AGENT
# =============================================================================

import time
import numpy as np
import pandas as pd


class FattyLiverAgent:

    # =========================================================================
    # INIT
    # =========================================================================

    def __init__(self, model_package):

        self.agent_id = "FattyLiverAgent"

        self.task_type = "fatty_liver_classification"

        self.modality = "clinical_tabular"

        self.model_name = "LightGBM Pipeline"

        self.target_name = "selector"

        # ---------------------------------------------------------------------
        # Expected training features
        # ---------------------------------------------------------------------

        self.feature_names = [
            "mcv",
            "alkphos",
            "sgpt",
            "sgot",
            "gammagt",
            "drinks"
        ]

        self.target_classes = None

        # ---------------------------------------------------------------------
        # Load model
        # ---------------------------------------------------------------------

        if isinstance(model_package, dict):

            if "model" not in model_package:

                raise ValueError(
                    "model_package dictionary must contain a 'model' key."
                )

            self.model = model_package["model"]

            if "target_classes" in model_package:

                self.target_classes = model_package[
                    "target_classes"
                ]

        else:

            # Accept sklearn Pipeline / estimator directly
            if hasattr(model_package, "predict"):

                self.model = model_package

            else:

                raise TypeError(
                    "FattyLiverAgent expects a fitted model/Pipeline "
                    "or a dictionary containing 'model'."
                )

        # ---------------------------------------------------------------------
        # Try to recover classes from the model
        # ---------------------------------------------------------------------

        if self.target_classes is None:

            if hasattr(self.model, "classes_"):

                self.target_classes = [
                    str(c)
                    for c in self.model.classes_
                ]

            elif hasattr(self.model, "named_steps"):

                for _, step in self.model.named_steps.items():

                    if hasattr(step, "classes_"):

                        self.target_classes = [
                            str(c)
                            for c in step.classes_
                        ]

                        break

        # ---------------------------------------------------------------------
        # Final fallback
        # ---------------------------------------------------------------------

        if self.target_classes is None:

            self.target_classes = [
                "0",
                "1",
                "2"
            ]

    # =========================================================================
    # INPUT PREPARATION
    # =========================================================================

    def _prepare_input(self, patient_data):

        # ---------------------------------------------------------------------
        # Convert input to DataFrame
        # ---------------------------------------------------------------------

        if isinstance(patient_data, pd.DataFrame):

            X = patient_data.copy()

        elif isinstance(patient_data, dict):

            X = pd.DataFrame(
                [patient_data]
            )

        else:

            raise TypeError(
                "patient_data must be a dictionary or pandas DataFrame."
            )

        # ---------------------------------------------------------------------
        # Remove target if accidentally provided
        # ---------------------------------------------------------------------

        if self.target_name in X.columns:

            X = X.drop(
                columns=[self.target_name]
            )

        # ---------------------------------------------------------------------
        # Add missing expected features
        # ---------------------------------------------------------------------

        for feature in self.feature_names:

            if feature not in X.columns:

                X[feature] = np.nan

        # ---------------------------------------------------------------------
        # Keep EXACT training order
        # ---------------------------------------------------------------------

        X = X.loc[
            :,
            self.feature_names
        ].copy()

        # ---------------------------------------------------------------------
        # Convert every feature explicitly to float
        #
        # This avoids integer/object dtype issues and makes the input
        # compatible with the preprocessing pipeline.
        # ---------------------------------------------------------------------

        for feature in self.feature_names:

            X[feature] = pd.to_numeric(
                X[feature],
                errors="coerce"
            ).astype(float)

        # ---------------------------------------------------------------------
        # Force DataFrame
        # ---------------------------------------------------------------------

        X = pd.DataFrame(
            X,
            columns=self.feature_names
        )

        return X

    # =========================================================================
    # PREDICT
    # =========================================================================

    def predict(self, patient_data):

        start_time = time.time()

        try:

            # -----------------------------------------------------------------
            # Prepare input
            # -----------------------------------------------------------------

            X = self._prepare_input(
                patient_data
            )

            # -----------------------------------------------------------------
            # Missing data quality
            # -----------------------------------------------------------------

            missing_ratio = float(
                X.isna().sum().sum()
                / X.size
            )

            quality = max(
                0.0,
                1.0 - missing_ratio
            )

            # -----------------------------------------------------------------
            # Prediction
            #
            # IMPORTANT:
            # The Pipeline performs its own preprocessing/imputation.
            # -----------------------------------------------------------------

            prediction = self.model.predict(
                X
            )

            # -----------------------------------------------------------------
            # Probability
            # -----------------------------------------------------------------

            probabilities = None

            if hasattr(
                self.model,
                "predict_proba"
            ):

                probabilities = self.model.predict_proba(
                    X
                )[0]

            # -----------------------------------------------------------------
            # Convert probabilities
            # -----------------------------------------------------------------

            if probabilities is not None:

                probabilities = np.asarray(
                    probabilities,
                    dtype=float
                )

                confidence = float(
                    np.max(probabilities)
                )

                uncertainty = float(
                    1.0 - confidence
                )

                # -------------------------------------------------------------
                # Ensure number of class labels matches probabilities
                # -------------------------------------------------------------

                if len(self.target_classes) != len(
                    probabilities
                ):

                    self.target_classes = [
                        str(i)
                        for i in range(
                            len(probabilities)
                        )
                    ]

                class_probabilities = {

                    self.target_classes[i]:
                    float(probabilities[i])

                    for i in range(
                        len(probabilities)
                    )
                }

            else:

                confidence = None

                uncertainty = None

                class_probabilities = {}

            # -----------------------------------------------------------------
            # Prediction value
            # -----------------------------------------------------------------

            prediction_value = prediction[0]

            # -----------------------------------------------------------------
            # Result
            # -----------------------------------------------------------------

            result = {

                "status": "success",

                "agent": self.agent_id,

                "agent_id": self.agent_id,

                "task_type": self.task_type,

                "modality": self.modality,

                "model": self.model_name,

                "prediction": str(
                    prediction_value
                ),

                "predicted_label": str(
                    prediction_value
                ),

                "probability": confidence,

                "confidence": confidence,

                "uncertainty": uncertainty,

                "quality": quality,

                "missing_ratio": missing_ratio,

                "missing_data_ratio": missing_ratio,

                "class_probabilities":
                    class_probabilities,

                "features_used":
                    list(self.feature_names),

                "inference_time":
                    time.time() - start_time,

                "latency_ms":
                    (
                        time.time() - start_time
                    ) * 1000,

                "error": None
            }

            return result

        # ---------------------------------------------------------------------
        # Error handling
        # ---------------------------------------------------------------------

        except Exception as e:

            return {

                "status": "error",

                "agent": self.agent_id,

                "agent_id": self.agent_id,

                "task_type": self.task_type,

                "modality": self.modality,

                "model": self.model_name,

                "prediction": None,

                "probability": None,

                "confidence": 0.0,

                "uncertainty": 1.0,

                "quality": 0.0,

                "missing_ratio": 1.0,

                "missing_data_ratio": 1.0,

                "class_probabilities": {},

                "features_used":
                    list(self.feature_names),

                "inference_time":
                    time.time() - start_time,

                "latency_ms":
                    (
                        time.time() - start_time
                    ) * 1000,

                "error": str(e)
            }

    # =========================================================================
    # ALIAS
    # =========================================================================

    def analyze(self, patient_data):

        return self.predict(
            patient_data
        )
