

import time
import numpy as np
import pandas as pd


class FattyLiverAgent:

    def __init__(self, model_package):

        self.name = "FattyLiverAgent"

        # The saved file is a complete sklearn Pipeline
        self.model = model_package

        self.model_name = "LightGBM Pipeline"

        self.feature_names = [
            "mcv",
            "alkphos",
            "sgpt",
            "sgot",
            "gammagt",
            "drinks"
        ]

        self.numerical_columns = self.feature_names.copy()
        self.categorical_columns = []

        self.target_name = "selector"

        # Pipeline -> final LightGBM classifier
        try:
            classifier = self.model.named_steps["classifier"]

            if hasattr(classifier, "classes_"):
                self.target_classes = [
                    str(x) for x in classifier.classes_
                ]
            else:
                self.target_classes = ["1", "2"]

        except Exception:
            self.target_classes = ["1", "2"]

    # =====================================================================
    # DATAFRAME
    # =====================================================================

    def _create_dataframe(self, patient_data):

        if isinstance(patient_data, dict):
            return pd.DataFrame([patient_data])

        if isinstance(patient_data, pd.DataFrame):
            return patient_data.copy()

        return pd.DataFrame(
            [patient_data],
            columns=self.feature_names
        )

    # =====================================================================
    # PREDICT
    # =====================================================================

    def predict(self, patient_data):

        start_time = time.perf_counter()

        try:

            X = self._create_dataframe(patient_data)

            # -------------------------------------------------------------
            # Remove target if provided
            # -------------------------------------------------------------

            if self.target_name in X.columns:
                X = X.drop(columns=[self.target_name])

            # -------------------------------------------------------------
            # Add missing features
            # -------------------------------------------------------------

            for feature in self.feature_names:

                if feature not in X.columns:
                    X[feature] = np.nan

            # -------------------------------------------------------------
            # Keep exact training order
            # -------------------------------------------------------------

            X = X[self.feature_names].copy()

            # -------------------------------------------------------------
            # Convert to numeric
            # -------------------------------------------------------------

            for feature in self.feature_names:

                X[feature] = pd.to_numeric(
                    X[feature],
                    errors="coerce"
                )

            # -------------------------------------------------------------
            # Missing ratio
            # -------------------------------------------------------------

            total_values = X.shape[0] * len(self.feature_names)

            missing_values = int(
                X.isna().sum().sum()
            )

            missing_ratio = float(
                missing_values /
                max(total_values, 1)
            )

            # -------------------------------------------------------------
            # PREDICTION
            #
            # IMPORTANT:
            # The Pipeline already contains:
            #
            # SimpleImputer -> LightGBM
            #
            # Therefore we do NOT impute manually.
            # -------------------------------------------------------------

            prediction = self.model.predict(X)[0]

            # -------------------------------------------------------------
            # PROBABILITY
            # -------------------------------------------------------------

            probabilities = None

            if hasattr(self.model, "predict_proba"):

                probabilities = (
                    self.model.predict_proba(X)[0]
                )

            # -------------------------------------------------------------
            # CONFIDENCE
            # -------------------------------------------------------------

            if probabilities is not None:

                confidence = float(
                    np.max(probabilities)
                )

                class_probabilities = {
                    str(class_name): float(probability)
                    for class_name, probability in zip(
                        self.target_classes,
                        probabilities
                    )
                }

            else:

                confidence = 0.0
                class_probabilities = None

            # -------------------------------------------------------------
            # UNCERTAINTY
            # -------------------------------------------------------------

            uncertainty = float(
                1.0 - confidence
            )

            # -------------------------------------------------------------
            # QUALITY
            # -------------------------------------------------------------

            quality = float(
                1.0 - missing_ratio
            )

            quality = max(
                0.0,
                min(1.0, quality)
            )

            # -------------------------------------------------------------
            # INFERENCE TIME
            # -------------------------------------------------------------

            inference_time = float(
                time.perf_counter() - start_time
            )

            # -------------------------------------------------------------
            # STANDARDIZED RESULT
            # -------------------------------------------------------------

            return {

                "status": "success",

                "agent": self.name,

                "model": self.model_name,

                "prediction": str(prediction),

                "confidence": confidence,

                "uncertainty": uncertainty,

                "quality": quality,

                "missing_ratio": missing_ratio,

                "class_probabilities": class_probabilities,

                "features_used": self.feature_names.copy(),

                "inference_time": inference_time
            }

        except Exception as e:

            inference_time = float(
                time.perf_counter() - start_time
            )

            return {

                "status": "error",

                "agent": self.name,

                "model": self.model_name,

                "prediction": None,

                "confidence": 0.0,

                "uncertainty": 1.0,

                "quality": 0.0,

                "missing_ratio": 1.0,

                "class_probabilities": None,

                "features_used": self.feature_names.copy(),

                "inference_time": inference_time,

                "error": str(e)
            }

    # =====================================================================
    # RUN
    # =====================================================================

    def run(self, patient_data):

        return self.predict(patient_data)

    # =====================================================================
    # TEST
    # =====================================================================

    def test(self):

        test_patient = {

            "mcv": 85.0,
            "alkphos": 85.0,
            "sgpt": 45.0,
            "sgot": 35.0,
            "gammagt": 50.0,
            "drinks": 5.0
        }

        result = self.predict(test_patient)

        print("=" * 70)
        print("FATTY LIVER AGENT TEST")
        print("=" * 70)

        print("\nStatus       :", result["status"])
        print("Prediction   :", result["prediction"])
        print("Confidence   :", result["confidence"])
        print("Uncertainty  :", result["uncertainty"])
        print("Quality      :", result["quality"])
        print("Missing ratio:", result["missing_ratio"])
        print("Probabilities:", result["class_probabilities"])
        print("Time         :", result["inference_time"])

        if result["status"] == "error":
            print("Error        :", result["error"])

        return result
