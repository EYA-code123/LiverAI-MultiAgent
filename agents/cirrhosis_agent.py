%%writefile /content/LiverAI-MultiAgent/agents/cirrhosis_agent.py

# ==========================================================
# CIRRHOSIS AGENT - CORRECTED VERSION
# ==========================================================

import pandas as pd
import numpy as np


class CirrhosisAgent:

    def __init__(self, package):

        self.name = "CirrhosisAgent"
        self.model_name = "XGBoost"

        # --------------------------------------------------
        # Load package
        # --------------------------------------------------

        self.package = package

        self.model = package["model"]

        self.feature_names = list(
            package["feature_names"]
        )

        self.numerical_columns = list(
            package["numerical_columns"]
        )

        self.categorical_columns = list(
            package["categorical_columns"]
        )

        self.encoders = package["encoders"]

        self.target_encoder = package["target_encoder"]

        self.numerical_imputer = (
            package["numerical_imputer"]
        )

        self.categorical_imputer = (
            package["categorical_imputer"]
        )

        # --------------------------------------------------
        # IMPORTANT FIX
        # --------------------------------------------------
        # Stage exists in the saved numerical imputer
        # but NOT in the XGBoost model features.
        #
        # Therefore Stage must NOT be sent to the model.
        # We only use the columns expected by the model.
        # --------------------------------------------------

        self.imputer_numerical_features = [
            col
            for col in self.numerical_columns
            if col in self.numerical_imputer.feature_names_in_
        ]

        print("=" * 70)
        print("CIRRHOSIS AGENT INITIALIZED")
        print("=" * 70)

        print("Model features:")
        print(self.feature_names)

        print("\nNumerical columns:")
        print(self.numerical_columns)

        print("\nCategorical columns:")
        print(self.categorical_columns)

        print("\nImputer numerical features:")
        print(self.imputer_numerical_features)

        print("\nStage in model features:",
              "Stage" in self.feature_names)

        print("Stage in numerical columns:",
              "Stage" in self.numerical_columns)

        print("=" * 70)


    # ======================================================
    # PREDICT
    # ======================================================

    def predict(self, patient_data):

        # --------------------------------------------------
        # Convert input to DataFrame
        # --------------------------------------------------

        if isinstance(patient_data, dict):

            X = pd.DataFrame([patient_data])

        elif isinstance(patient_data, pd.DataFrame):

            X = patient_data.copy()

        else:

            X = pd.DataFrame(
                [patient_data],
                columns=self.feature_names
            )


        # --------------------------------------------------
        # Check required model features
        # --------------------------------------------------

        missing_features = [
            col
            for col in self.feature_names
            if col not in X.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing cirrhosis features: "
                + str(missing_features)
            )


        # --------------------------------------------------
        # Keep ONLY model features
        # --------------------------------------------------

        X = X[self.feature_names].copy()


        # ==================================================
        # NUMERICAL PREPROCESSING
        # ==================================================

        # IMPORTANT:
        # Do NOT use all numerical_columns here.
        #
        # Stage is present in numerical_columns but absent
        # from feature_names and therefore absent from X.
        #
        # The imputer expects Stage because it was fitted with it.
        #
        # We therefore create a temporary DataFrame containing
        # exactly the columns expected by the imputer.

        imputer_columns = list(
            self.numerical_imputer.feature_names_in_
        )

        X_imputer = pd.DataFrame(
            index=X.index
        )

        for col in imputer_columns:

            if col in X.columns:

                X_imputer[col] = X[col]

            else:

                # Stage is not a model feature.
                # Use NaN so the saved imputer can fill it.
                X_imputer[col] = np.nan


        # --------------------------------------------------
        # Apply saved numerical imputer
        # --------------------------------------------------

        X_imputed = self.numerical_imputer.transform(
            X_imputer
        )

        X_imputed = pd.DataFrame(
            X_imputed,
            columns=imputer_columns,
            index=X.index
        )


        # --------------------------------------------------
        # Copy imputed values ONLY for model features
        # --------------------------------------------------

        for col in self.feature_names:

            if col in X_imputed.columns:

                X[col] = X_imputed[col]


        # ==================================================
        # CATEGORICAL PREPROCESSING
        # ==================================================

        categorical_features = [
            col
            for col in self.categorical_columns
            if col in self.feature_names
        ]


        if categorical_features:

            X_cat = X[
                categorical_features
            ].copy()

            X_cat = self.categorical_imputer.transform(
                X_cat
            )

            X_cat = pd.DataFrame(
                X_cat,
                columns=categorical_features,
                index=X.index
            )

            X[categorical_features] = X_cat


        # ==================================================
        # APPLY SAVED LABEL ENCODERS
        # ==================================================

        for col in categorical_features:

            if col not in self.encoders:
                continue

            encoder = self.encoders[col]

            values = X[col].astype(str)

            known_values = set(
                encoder.classes_
            )

            # Handle unknown categories
            values = values.apply(
                lambda x:
                x if x in known_values
                else encoder.classes_[0]
            )

            X[col] = encoder.transform(
                values
            )


        # ==================================================
        # FINAL COLUMN ORDER
        # ==================================================

        X = X[
            self.feature_names
        ].copy()


        # ==================================================
        # PREDICTION
        # ==================================================

        prediction_encoded = self.model.predict(
            X
        )[0]


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
                .predict_proba(X)[0]
            )

            probability = float(
                np.max(probabilities)
            )


        # ==================================================
        # DECODE PREDICTION
        # ==================================================

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
