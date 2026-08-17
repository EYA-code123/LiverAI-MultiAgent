%%writefile /content/LiverAI-MultiAgent/agents/cirrhosis_agent.py

import numpy as np
import pandas as pd


class CirrhosisAgent:

    def __init__(self, package):

        self.name = "CirrhosisAgent"
        self.model_name = "XGBoost"

        self.model = package["model"]

        # Features réellement utilisées par XGBoost
        self.feature_names = list(
            package["feature_names"]
        )

        # Colonnes numériques sauvegardées
        saved_numerical = list(
            package["numerical_columns"]
        )

        # IMPORTANT :
        # Stage est dans l'imputer historique
        # mais PAS dans le modèle XGBoost.
        self.numerical_columns = [
            col
            for col in saved_numerical
            if col in self.feature_names
        ]

        # Colonnes catégorielles
        saved_categorical = list(
            package["categorical_columns"]
        )

        self.categorical_columns = [
            col
            for col in saved_categorical
            if col in self.feature_names
        ]

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

        print("=" * 70)
        print("CIRRHOSIS AGENT INITIALIZED")
        print("=" * 70)

        print("Model features:")
        print(self.feature_names)

        print("\nNumerical features:")
        print(self.numerical_columns)

        print("\nCategorical features:")
        print(self.categorical_columns)

        if self.numerical_imputer is not None:

            print("\nImputer features:")
            print(
                list(
                    self.numerical_imputer
                    .feature_names_in_
                )
            )

        print("=" * 70)


    def predict(self, patient_data):

        # ==========================================================
        # 1. DATAFRAME
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
        # 2. AJOUT DES FEATURES MANQUANTES
        # ==========================================================

        for col in self.feature_names:

            if col not in X.columns:
                X[col] = np.nan

        # ==========================================================
        # 3. GARDER UNIQUEMENT LES 18 FEATURES DU MODELE
        # ==========================================================

        X = X[
            self.feature_names
        ].copy()

        # ==========================================================
        # 4. NUMERICAL IMPUTATION
        # ==========================================================

        if self.numerical_imputer is not None:

            # Colonnes attendues par l'ancien imputer
            imputer_features = list(
                self.numerical_imputer
                .feature_names_in_
            )

            # Construire EXACTEMENT les colonnes
            # attendues par l'imputer
            numerical_input = pd.DataFrame(
                index=X.index
            )

            for col in imputer_features:

                if col in X.columns:

                    numerical_input[col] = pd.to_numeric(
                        X[col],
                        errors="coerce"
                    )

                else:

                    # Stage n'est pas fourni par le patient.
                    # On utilise la statistique sauvegardée
                    # par l'imputer.
                    idx = imputer_features.index(
                        col
                    )

                    numerical_input[col] = (
                        self.numerical_imputer
                        .statistics_[idx]
                    )

            # Transformation
            imputed = (
                self.numerical_imputer
                .transform(
                    numerical_input
                )
            )

            imputed_df = pd.DataFrame(
                imputed,
                columns=imputer_features,
                index=X.index
            )

            # Remettre seulement les colonnes
            # numériques du modèle
            for col in self.numerical_columns:

                X[col] = (
                    imputed_df[col]
                    .values
                )

        # ==========================================================
        # 5. CATEGORICAL IMPUTATION
        # ==========================================================

        if (
            self.categorical_imputer is not None
            and self.categorical_columns
        ):

            imputer_features = list(
                self.categorical_imputer
                .feature_names_in_
            )

            categorical_input = pd.DataFrame(
                index=X.index
            )

            for col in imputer_features:

                if col in X.columns:
                    categorical_input[col] = (
                        X[col].astype(object)
                    )
                else:
                    categorical_input[col] = np.nan

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

            for col in self.categorical_columns:

                if col in imputed_df.columns:

                    X[col] = (
                        imputed_df[col]
                        .values
                    )

        # ==========================================================
        # 6. ENCODAGE
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
        # 7. ORDRE FINAL DES FEATURES
        # ==========================================================

        X = X[
            self.feature_names
        ].copy()

        # ==========================================================
        # 8. XGBOOST PREDICTION
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
            "prediction": str(prediction),
            "probability": probability,
            "status": "completed"
        }
