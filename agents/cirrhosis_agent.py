%%writefile /content/LiverAI-MultiAgent/agents/cirrhosis_agent.py

# ==========================================================
# LiverAI - Cirrhosis Agent
# ==========================================================

import numpy as np
import pandas as pd


class CirrhosisAgent:

    def __init__(self, package):

        self.name = "CirrhosisAgent"
        self.model_name = "XGBoost"

        # ==================================================
        # LOAD PACKAGE
        # ==================================================

        self.package = package

        self.model = package["model"]

        # ==================================================
        # FEATURES
        # ==================================================

        self.feature_names = [
            "N_Days",
            "Status",
            "Drug",
            "Age",
            "Sex",
            "Ascites",
            "Hepatomegaly",
            "Spiders",
            "Edema",
            "Bilirubin",
            "Cholesterol",
            "Albumin",
            "Copper",
            "Alk_Phos",
            "SGOT",
            "Tryglicerides",
            "Platelets",
            "Prothrombin"
        ]

        # ==================================================
        # NUMERICAL FEATURES
        # IMPORTANT:
        # Stage is NEVER used here
        # ==================================================

        self.numerical_columns = [
            "N_Days",
            "Age",
            "Bilirubin",
            "Cholesterol",
            "Albumin",
            "Copper",
            "Alk_Phos",
            "SGOT",
            "Tryglicerides",
            "Platelets",
            "Prothrombin"
        ]

        # ==================================================
        # CATEGORICAL FEATURES
        # ==================================================

        self.categorical_columns = [
            "Status",
            "Drug",
            "Sex",
            "Ascites",
            "Hepatomegaly",
            "Spiders",
            "Edema"
        ]

        # ==================================================
        # PREPROCESSING OBJECTS
        # ==================================================

        self.encoders = package.get(
            "encoders",
            {}
        )

        self.target_encoder = package.get(
            "target_encoder",
            None
        )

        self.numerical_imputer = package.get(
            "numerical_imputer",
            None
        )

        self.categorical_imputer = package.get(
            "categorical_imputer",
            None
        )

        print("✓ CirrhosisAgent initialized")

        print(
            "Numerical features:",
            self.numerical_columns
        )

        print(
            "Categorical features:",
            self.categorical_columns
        )

    # ======================================================
    # PREDICT
    # ======================================================

    def predict(self, patient_data):

        # ==================================================
        # CONVERT INPUT TO DATAFRAME
        # ==================================================

        if isinstance(patient_data, dict):

            X = pd.DataFrame(
                [patient_data]
            )

        elif isinstance(
            patient_data,
            pd.DataFrame
        ):

            X = patient_data.copy()

        else:

            X = pd.DataFrame(
                [patient_data],
                columns=self.feature_names
            )

        # ==================================================
        # CHECK REQUIRED FEATURES
        # ==================================================

        missing_features = [
            col
            for col in self.feature_names
            if col not in X.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing Cirrhosis features: "
                + str(missing_features)
            )

        # ==================================================
        # KEEP ONLY MODEL FEATURES
        # ==================================================

        X = X[
            self.feature_names
        ].copy()

        # ==================================================
        # NUMERICAL PREPROCESSING
        # ==================================================

        numerical_X = X[
            self.numerical_columns
        ].copy()

        # --------------------------------------------------
        # IMPORTANT FIX
        # Do NOT use an imputer trained with Stage
        # --------------------------------------------------

        if self.numerical_imputer is not None:

            expected_features = getattr(
                self.numerical_imputer,
                "feature_names_in_",
                None
            )

            # If old imputer contains Stage,
            # use manual median filling instead.
            if (
                expected_features is not None
                and "Stage" in expected_features
            ):

                print(
                    "⚠️ Old imputer contains Stage."
                )

                print(
                    "→ Using safe numerical preprocessing."
                )

                numerical_X = (
                    numerical_X.apply(
                        pd.to_numeric,
                        errors="coerce"
                    )
                )

                numerical_X = (
                    numerical_X.fillna(
                        numerical_X.median()
                    )
                )

            else:

                numerical_X = (
                    self.numerical_imputer.transform(
                        numerical_X
                    )
                )

                numerical_X = pd.DataFrame(
                    numerical_X,
                    columns=self.numerical_columns,
                    index=X.index
                )

        else:

            numerical_X = (
                numerical_X.apply(
                    pd.to_numeric,
                    errors="coerce"
                )
            )

            numerical_X = (
                numerical_X.fillna(
                    numerical_X.median()
                )
            )

        # ==================================================
        # CATEGORICAL PREPROCESSING
        # ==================================================

        categorical_X = X[
            self.categorical_columns
        ].copy()

        if self.categorical_imputer is not None:

            categorical_X = (
                self.categorical_imputer.transform(
                    categorical_X
                )
            )

            categorical_X = pd.DataFrame(
                categorical_X,
                columns=self.categorical_columns,
                index=X.index
            )

        else:

            categorical_X = (
                categorical_X.fillna(
                    "Unknown"
                )
            )

        # ==================================================
        # APPLY LABEL ENCODERS
        # ==================================================

        for col in self.categorical_columns:

            if col in self.encoders:

                encoder = self.encoders[col]

                values = (
                    categorical_X[col]
                    .astype(str)
                )

                known_values = set(
                    encoder.classes_
                )

                values = values.apply(
                    lambda x:
                    x
                    if x in known_values
                    else encoder.classes_[0]
                )

                categorical_X[col] = (
                    encoder.transform(values)
                )

            else:

                categorical_X[col] = (
                    pd.to_numeric(
                        categorical_X[col],
                        errors="coerce"
                    ).fillna(0)
                )

        # ==================================================
        # COMBINE FEATURES
        # ==================================================

        X_processed = pd.concat(
            [
                numerical_X,
                categorical_X
            ],
            axis=1
        )

        # ==================================================
        # RESTORE EXACT MODEL ORDER
        # ==================================================

        X_processed = X_processed[
            self.feature_names
        ]

        # ==================================================
        # FINAL SAFETY CHECK
        # ==================================================

        if "Stage" in X_processed.columns:

            raise RuntimeError(
                "ERROR: Stage detected in prediction features."
            )

        # ==================================================
        # PREDICTION
        # ==================================================

        prediction_encoded = (
            self.model.predict(
                X_processed
            )[0]
        )

        # ==================================================
        # PROBABILITY
        # ==================================================

        probability = None

        if hasattr(
            self.model,
            "predict_proba"
        ):

            probabilities = (
                self.model
                .predict_proba(
                    X_processed
                )[0]
            )

            probability = float(
                np.max(probabilities)
            )

        # ==================================================
        # DECODE PREDICTION
        # ==================================================

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

        # ==================================================
        # RETURN
        # ==================================================

        return {

            "agent": self.name,

            "model": self.model_name,

            "prediction": str(
                prediction
            ),

            "probability": probability,

            "status": "completed"
        }
