%%writefile /content/LiverAI-MultiAgent/agents/cirrhosis_agent.py

import numpy as np
import pandas as pd


class CirrhosisAgent:

    def __init__(self, package):

        self.name = "CirrhosisAgent"
        self.model_name = "XGBoost"

        # ==========================================================
        # MODEL
        # ==========================================================

        self.model = package["model"]

        # ==========================================================
        # FEATURES DU MODELE
        # ==========================================================

        self.feature_names = list(
            package["feature_names"]
        )

        # ==========================================================
        # COLONNES NUMERIQUES
        # ==========================================================

        saved_numerical = list(
            package["numerical_columns"]
        )

        # IMPORTANT :
        # Stage est présent dans l'imputer mais PAS dans le modèle.
        # On le retire.

        self.numerical_columns = [
            col
            for col in saved_numerical
            if col in self.feature_names
        ]

        # ==========================================================
        # COLONNES CATEGORIELLES
        # ==========================================================

        saved_categorical = list(
            package["categorical_columns"]
        )

        self.categorical_columns = [
            col
            for col in saved_categorical
            if col in self.feature_names
        ]

        # ==========================================================
        # ENCODERS
        # ==========================================================

        self.encoders = package.get(
            "encoders",
            {}
        )

        # ==========================================================
        # TARGET ENCODER
        # ==========================================================

        self.target_encoder = package.get(
            "target_encoder",
            None
        )

        # ==========================================================
        # IMPUTERS
        # ==========================================================

        self.numerical_imputer = package.get(
            "numerical_imputer",
            None
        )

        self.categorical_imputer = package.get(
            "categorical_imputer",
            None
        )

        print("=" * 70)
        print("CIRRHOSIS AGENT INITIALIZED")
        print("=" * 70)

        print("Model features:")
        print(self.feature_names)

        print("\nNumerical features:")
        print(self.numerical_columns)

        print("\nCategorical features:")
        print(self.categorical_columns)

        print("\nIgnored preprocessing columns:")
        ignored = [
            col
            for col in saved_numerical
            if col not in self.feature_names
        ]
        print(ignored)

        print("=" * 70)


    # ==============================================================
    # PREDICT
    # ==============================================================

    def predict(self, patient_data):

        # ==========================================================
        # 1. CONVERT TO DATAFRAME
        # ==========================================================

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

        # ==========================================================
        # 2. ADD MISSING MODEL FEATURES
        # ==========================================================

        for col in self.feature_names:

            if col not in X.columns:

                X[col] = np.nan

        # ==========================================================
        # 3. KEEP ONLY MODEL FEATURES
        # ==========================================================

        X = X[
            self.feature_names
        ].copy()

        # ==========================================================
        # 4. NUMERICAL FEATURES
        # ==========================================================

        if self.numerical_columns:

            numerical_data = X[
                self.numerical_columns
            ].copy()

            # Convert to numeric

            for col in self.numerical_columns:

                numerical_data[col] = pd.to_numeric(
                    numerical_data[col],
                    errors="coerce"
                )

            # ------------------------------------------------------
            # IMPORTANT
            # ------------------------------------------------------
            # The saved imputer expects 12 columns including Stage.
            #
            # We therefore construct a temporary dataframe with
            # exactly the columns expected by the imputer.
            # Stage is filled with the saved median.
            # It is then discarded after imputation.
            # ------------------------------------------------------

            if self.numerical_imputer is not None:

                imputer_features = list(
                    self.numerical_imputer.feature_names_in_
                )

                imputer_input = pd.DataFrame(
                    index=X.index
                )

                for col in imputer_features:

                    if col in numerical_data.columns:

                        imputer_input[col] = (
                            numerical_data[col]
                        )

                    else:

                        # Stage is not a model input.
                        # Use the imputer's own median.

                        index = imputer_features.index(
                            col
                        )

                        imputer_input[col] = (
                            self.numerical_imputer
                            .statistics_[index]
                        )

                imputed = (
                    self.numerical_imputer
                    .transform(imputer_input)
                )

                imputed_df = pd.DataFrame(
                    imputed,
                    columns=imputer_features,
                    index=X.index
                )

                # Keep ONLY model numerical features

                for col in self.numerical_columns:

                    X[col] = imputed_df[col].values

        # ==========================================================
        # 5. CATEGORICAL IMPUTATION
        # ==========================================================

        if self.categorical_columns:

            categorical_data = X[
                self.categorical_columns
            ].copy()

            categorical_data = (
                categorical_data.astype(object)
            )

            if self.categorical_imputer is not None:

                imputer_features = list(
                    self.categorical_imputer
                    .feature_names_in_
                )

                # Keep only features actually used
                # by the model

                categorical_input = (
                    categorical_data[
                        imputer_features
                    ]
                )

                imputed = (
                    self.categorical_imputer
                    .transform(
                        categorical_input
                    )
                )

                imputed_df = pd.DataFrame(
                    imputed,
                    columns=imputer_features,
                    index=X.index
                )

                for col in imputer_features:

                    X[col] = (
                        imputed_df[col]
                        .values
                    )

        # ==========================================================
        # 6. ENCODE CATEGORICAL FEATURES
        # ==========================================================

        for col in self.categorical_columns:

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

            # Unknown category handling

            values = values.apply(
                lambda x:
                x
                if x in known_values
                else encoder.classes_[0]
            )

            X[col] = encoder.transform(
                values
            )

        # ==========================================================
        # 7. FINAL COLUMN ORDER
        # ==========================================================

        X = X[
            self.feature_names
        ].copy()

        # ==========================================================
        # 8. PREDICTION
        # ==========================================================

        prediction_encoded = (
            self.model
            .predict(X)[0]
        )

        # ==========================================================
        # 9. PROBABILITY
        # ==========================================================

        probability = None

        if hasattr(
            self.model,
            "predict_proba"
        ):

            probabilities = (
                self.model
                .predict_proba(X)[0]
            )

            probability = float(
                np.max(probabilities)
            )

        # ==========================================================
        # 10. DECODE TARGET
        # ==========================================================

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

        # ==========================================================
        # 11. RESULT
        # ==========================================================

        return {

            "agent": self.name,

            "model": self.model_name,

            "prediction": str(
                prediction
            ),

            "probability": probability,

            "status": "completed"

        }
