%%writefile /content/LiverAI-MultiAgent/agents/clinical_reasoning_agent.py

import os
import numpy as np
import pandas as pd

from pytorch_tabular import TabularModel


class ClinicalReasoningAgent:

    FEATURE_COLUMNS = [
        "mcv",
        "alkphos",
        "sgpt",
        "sgot",
        "gammagt",
        "drinks"
    ]

    def __init__(self, model_package):

        self.name = "ClinicalReasoningAgent"

        # ============================================================
        # VALIDATION DU CHEMIN DU MODÈLE
        # ============================================================

        if model_package is None:
            raise ValueError(
                "ClinicalReasoningAgent: model_package cannot be None."
            )

        if not isinstance(model_package, (str, os.PathLike)):
            raise TypeError(
                "ClinicalReasoningAgent: model_package must be "
                "a path to the TabTransformer model directory."
            )

        self.model_path = os.fspath(model_package)

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"ClinicalReasoningAgent: model directory not found:\n"
                f"{self.model_path}"
            )

        # ============================================================
        # CHARGEMENT DU TABTRANSFORMER
        # ============================================================

        print(
            f"Loading Clinical Reasoning model from:\n"
            f"{self.model_path}"
        )

        self.model = TabularModel.load_model(self.model_path)

        self.model_name = "TabTransformer"

        # ============================================================
        # FEATURES
        # ============================================================

        self.feature_names = self.FEATURE_COLUMNS.copy()

        self.numerical_columns = self.FEATURE_COLUMNS.copy()

        self.categorical_columns = []

        # ============================================================
        # TARGET
        # ============================================================

        self.target_name = "selector"

        self.target_classes = [0, 1]

        print("✓ ClinicalReasoningAgent initialized")
        print(f"✓ Model: {self.model_name}")
        print(f"✓ Features: {self.feature_names}")

    # ================================================================
    # CREATE DATAFRAME
    # ================================================================

    def _create_dataframe(self, patient_data):

        # ------------------------------------------------------------
        # Déjà un DataFrame
        # ------------------------------------------------------------

        if isinstance(patient_data, pd.DataFrame):

            df = patient_data.copy()

        # ------------------------------------------------------------
        # Dictionnaire
        # ------------------------------------------------------------

        elif isinstance(patient_data, dict):

            df = pd.DataFrame([patient_data])

        # ------------------------------------------------------------
        # Liste / tuple / numpy array
        # ------------------------------------------------------------

        elif isinstance(patient_data, (list, tuple, np.ndarray)):

            values = np.asarray(patient_data)

            if values.ndim == 1:

                if len(values) != len(self.FEATURE_COLUMNS):
                    raise ValueError(
                        f"Expected {len(self.FEATURE_COLUMNS)} clinical "
                        f"features, got {len(values)}."
                    )

                df = pd.DataFrame(
                    [values],
                    columns=self.FEATURE_COLUMNS
                )

            elif values.ndim == 2:

                if values.shape[1] != len(self.FEATURE_COLUMNS):
                    raise ValueError(
                        f"Expected {len(self.FEATURE_COLUMNS)} columns, "
                        f"got {values.shape[1]}."
                    )

                df = pd.DataFrame(
                    values,
                    columns=self.FEATURE_COLUMNS
                )

            else:

                raise ValueError(
                    "patient_data must be 1D or 2D."
                )

        else:

            raise TypeError(
                "patient_data must be a pandas DataFrame, dict, "
                "list, tuple, or numpy array."
            )

        # ============================================================
        # VÉRIFICATION DES FEATURES
        # ============================================================

        missing = [
            col for col in self.FEATURE_COLUMNS
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing clinical features: "
                + ", ".join(missing)
            )

        # ============================================================
        # GARDER UNIQUEMENT LES FEATURES DU MODÈLE
        # ============================================================

        df = df[self.FEATURE_COLUMNS].copy()

        # ============================================================
        # CONVERSION NUMÉRIQUE
        # ============================================================

        for col in self.FEATURE_COLUMNS:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # ============================================================
        # VÉRIFICATION DES VALEURS MANQUANTES
        # ============================================================

        if df.isnull().any().any():

            missing_values = df.columns[
                df.isnull().any()
            ].tolist()

            raise ValueError(
                "Missing or invalid values detected in: "
                + ", ".join(missing_values)
            )

        return df

    # ================================================================
    # PREDICT
    # ================================================================

    def predict(self, patient_data):

        # ------------------------------------------------------------
        # Préparation
        # ------------------------------------------------------------

        df = self._create_dataframe(patient_data)

        # ------------------------------------------------------------
        # Prédiction TabTransformer
        # ------------------------------------------------------------

        prediction = self.model.predict(df)

        # ------------------------------------------------------------
        # Identifier les colonnes générées
        # ------------------------------------------------------------

        prediction_columns = [
            col
            for col in prediction.columns
            if "prediction" in col.lower()
        ]

        probability_columns = [
            col
            for col in prediction.columns
            if "probability" in col.lower()
        ]

        # ------------------------------------------------------------
        # Prediction
        # ------------------------------------------------------------

        if prediction_columns:

            predicted_class = int(
                prediction[prediction_columns[0]].iloc[0]
            )

        else:

            raise RuntimeError(
                "TabTransformer prediction column not found."
            )

        # ------------------------------------------------------------
        # Probabilités
        # ------------------------------------------------------------

        probabilities = {}

        for col in probability_columns:

            try:
                value = float(prediction[col].iloc[0])
            except Exception:
                continue

            probabilities[col] = value

        # ------------------------------------------------------------
        # Résultat
        # ------------------------------------------------------------

        result = {
            "agent": self.name,
            "model": self.model_name,
            "prediction": predicted_class,
            "probabilities": probabilities,
            "status": "success"
        }

        return result

    # ================================================================
    # ANALYZE
    # ================================================================

    def analyze(self, patient_data):

        return self.predict(patient_data)
