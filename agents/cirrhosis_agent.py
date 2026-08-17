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
        # PACKAGE
        # ==================================================

        self.package = package

        self.model = package["model"]

        # ==================================================
        # MODEL FEATURES
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
        # IMPORTANT: Stage is NOT included
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
        # ENCODERS
        # ==================================================

        self.encoders = package.get(
            "encoders",
            {}
        )

        self.target_encoder = package.get(
            "target_encoder",
            None
        )

        # ==================================================
        # OLD IMPUTERS
        # ==================================================

        self.old_numerical_imputer = package.get(
            "numerical_imputer",
            None
        )

        self.categorical_imputer = package.get(
            "categorical_imputer",
            None
        )

        print("✓ CirrhosisAgent initialized")

    # ======================================================
    # NUMERICAL IMPUTATION
    # ======================================================

    def _impute_numerical(self, X):

        X = X.copy()

        # Convert to numeric
        for col in self.numerical_columns:

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            )

        # --------------------------------------------------
        # IMPORTANT:
        # Never call old_imputer.transform()
        # because it expects Stage.
        # --------------------------------------------------

        old_imputer = self.old_numerical_imputer

        if old_imputer is not None:

            statistics = getattr(
                old_imputer,
                "statistics_",
                None
            )

            feature_names = getattr(
                old_imputer,
                "feature_names_in_",
                None
            )

            if (
                statistics is not None
                and feature_names is not None
            ):

                statistics_dict = dict(
                    zip(
                        feature_names,
                        statistics
                    )
                )

                for col in self.numerical_columns:

                    if col in statistics_dict:

                        X[col] = X[col].fillna(
                            statistics_dict[col]
                        )

                    else:

                        X[col] = X[col].fillna(
                            X[col].median()
                        )

            else:

                X = X.fillna(
                    X.median()
                )

        else:

            X = X.fillna(
                X.median()
            )

        return X

    # ======================================================
    # CATEGORICAL IMPUTATION
    # ======================================================

    def _impute_categorical(self, X):

        X = X.copy()

        if self.categorical_imputer is not None:

            expected = getattr(
                self.categorical_imputer,
                "feature_names_in_",
                None
            )

            if expected is not None:

                # Use only columns expected by imputer
                expected = [
                    col
                    for col in expected
                    if col in self.categorical_columns
                ]

                if expected:

                    temp = X[expected].copy()

                    temp = (
                        self.categorical_imputer
                        .transform(temp)
                    )

                    temp = pd.DataFrame(
                        temp,
                        columns=expected,
                        index=X.index
                    )

                    for col in expected:
                        X[col] = temp[col]

        # Final fallback
        for col in self.categorical_columns:

            X[col] = X[col].fillna(
                "Unknown"
            )

        return X

    # ======================================================
    # PREDICT
    # ======================================================

    def predict(self, patient_data):

        # ==================================================
        # INPUT → DATAFRAME
        # ==================================================

        if isinstance(
            patient_data,
            dict
        ):

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
        # CHECK FEATURES
        # ==================================================

        missing = [
            col
            for col in self.feature_names
            if col not in X.columns
        ]

        if missing:

            raise ValueError(
                "Missing Cirrhosis features: "
                + str(missing)
            )

        # ==================================================
        # KEEP ONLY INPUT FEATURES
        # ==================================================

        X = X[
            self.feature_names
        ].copy()

        # ==================================================
        # NUMERICAL IMPUTATION
        # ==================================================

        numerical_X = self._impute_numerical(
            X[self.numerical_columns]
        )

        # ==================================================
        # CATEGORICAL IMPUTATION
        # ==================================================

        categorical_X = self._impute_categorical(
            X[self.categorical_columns]
        )

        # ==================================================
        # ENCODE CATEGORICAL VARIABLES
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
                    lambda value:
                    value
                    if value in known_values
                    else encoder.classes_[0]
                )

                categorical_X[col] = (
                    encoder.transform(values)
                )

            else:

                categorical_X[col] = pd.to_numeric(
                    categorical_X[col],
                    errors="coerce"
                ).fillna(0)

        # ==================================================
        # COMBINE
        # ==================================================

        X_processed = pd.concat(
            [
                numerical_X,
                categorical_X
            ],
            axis=1
        )

        # ==================================================
        # EXACT FEATURE ORDER
        # ==================================================

        X_processed = X_processed[
            self.feature_names
        ]

        # ==================================================
        # SAFETY CHECK
        # ==================================================

        if "Stage" in X_processed.columns:

            raise RuntimeError(
                "Stage must NEVER be used "
                "as a prediction feature."
            )

        # ==================================================
        # PREDICT
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
        # DECODE TARGET
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
        # RESULT
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
