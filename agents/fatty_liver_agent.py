%%writefile /content/LiverAI-MultiAgent/agents/fatty_liver_agent.py

import time
import numpy as np
import pandas as pd


class FattyLiverAgent:

    def __init__(self, model_package):

        # ---------------------------------------------------------
        # Accept either:
        #   - sklearn Pipeline
        #   - dictionary containing a model
        # ---------------------------------------------------------
        if hasattr(model_package, "predict"):
            # Direct sklearn Pipeline / estimator
            self.model = model_package
            self.model_name = "LightGBM Pipeline"

        elif isinstance(model_package, dict):
            # Backward compatibility with old model-package format
            if "model" not in model_package:
                raise ValueError(
                    "Dictionary model_package must contain a 'model' key."
                )

            self.model = model_package["model"]
            self.model_name = model_package.get(
                "model_name",
                "LightGBM"
            )

        else:
            raise TypeError(
                "model_package must be either a sklearn model/Pipeline "
                "or a dictionary containing 'model'. "
                f"Received: {type(model_package)}"
            )

        # ---------------------------------------------------------
        # Features used by the trained Fatty Liver model
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Get classes from sklearn classifier
        # ---------------------------------------------------------
        self.target_classes = None

        try:
            if hasattr(self.model, "classes_"):
                self.target_classes = [
                    str(x) for x in self.model.classes_
                ]

            elif hasattr(self.model, "named_steps"):
                classifier = self.model.named_steps.get("classifier")

                if classifier is not None and hasattr(
                    classifier, "classes_"
                ):
                    self.target_classes = [
                        str(x) for x in classifier.classes_
                    ]

        except Exception:
            self.target_classes = None

    # =============================================================
    # PREDICTION
    # =============================================================
    def predict(self, patient_data):

        start_time = time.time()

        try:

            # -----------------------------------------------------
            # Convert input to DataFrame
            # -----------------------------------------------------
            if isinstance(patient_data, pd.DataFrame):

                X = patient_data.copy()

            elif isinstance(patient_data, dict):

                X = pd.DataFrame([patient_data])

            else:

                raise TypeError(
                    "patient_data must be a dictionary or pandas DataFrame."
                )

            # -----------------------------------------------------
            # Remove target if accidentally provided
            # -----------------------------------------------------
            if self.target_name in X.columns:

                X = X.drop(
                    columns=[self.target_name]
                )

            # -----------------------------------------------------
            # Make sure all expected features exist
            # -----------------------------------------------------
            for feature in self.feature_names:

                if feature not in X.columns:

                    X[feature] = np.nan

            # -----------------------------------------------------
            # Keep ONLY training features and correct order
            # -----------------------------------------------------
            X = X[self.feature_names].copy()

            # -----------------------------------------------------
            # Convert numerical features
            # -----------------------------------------------------
            for feature in self.feature_names:

                X[feature] = pd.to_numeric(
                    X[feature],
                    errors="coerce"
                )

            # -----------------------------------------------------
            # Missing data quality
            # -----------------------------------------------------
            missing_ratio = float(
                X.isna().sum().sum()
                / X.size
            )

            quality = max(
                0.0,
                1.0 - missing_ratio
            )

            # -----------------------------------------------------
            # IMPORTANT:
            # The Pipeline already contains its own imputer.
            # Therefore DO NOT manually impute here.
            # -----------------------------------------------------
            prediction = self.model.predict(X)

            # -----------------------------------------------------
            # Probabilities
            # -----------------------------------------------------
            if hasattr(self.model, "predict_proba"):

                probabilities = self.model.predict_proba(X)[0]

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

                if self.target_classes is None:

                    self.target_classes = [
                        str(i)
                        for i in range(len(probabilities))
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

            # -----------------------------------------------------
            # Prediction value
            # -----------------------------------------------------
            prediction_value = prediction[0]

            result = {

                "status": "success",

                "agent": "FattyLiverAgent",

                "model": self.model_name,

                "prediction": str(
                    prediction_value
                ),

                "confidence": confidence,

                "uncertainty": uncertainty,

                "quality": quality,

                "missing_ratio": missing_ratio,

                "class_probabilities":
                    class_probabilities,

                "features_used":
                    self.feature_names,

                "inference_time":
                    time.time() - start_time
            }

            return result

        except Exception as e:

            return {

                "status": "error",

                "agent": "FattyLiverAgent",

                "model": self.model_name,

                "error": str(e),

                "inference_time":
                    time.time() - start_time
            }

    # =============================================================
    # RUN ALIAS
    # =============================================================
    def run(self, patient_data):

        return self.predict(patient_data)

    # =============================================================
    # TECHNICAL TEST
    # =============================================================
    def test(self):

        test_patient = {

            "mcv": 85.0,

            "alkphos": 85.0,

            "sgpt": 45.0,

            "sgot": 35.0,

            "gammagt": 50.0,

            "drinks": 5.0
        }

        result = self.predict(
            test_patient
        )

        print("=" * 60)

        print(
            "FATTY LIVER AGENT TEST"
        )

        print("=" * 60)

        print(
            "Status:",
            result.get("status")
        )

        print(
            "Model:",
            result.get("model")
        )

        print(
            "Prediction:",
            result.get("prediction")
        )

        print(
            "Confidence:",
            result.get("confidence")
        )

        print(
            "Uncertainty:",
            result.get("uncertainty")
        )

        print(
            "Quality:",
            result.get("quality")
        )

        print(
            "Missing ratio:",
            result.get("missing_ratio")
        )

        print(
            "Class probabilities:",
            result.get("class_probabilities")
        )

        print("=" * 60)

        return result
