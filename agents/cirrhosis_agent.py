# =============================================================================
# 🔧 CORRECTION COMPLÈTE — CIRRHOSIS AGENT
# =============================================================================

from pathlib import Path

agent_file = Path("/content/agents/cirrhosis_agent.py")

code = r'''
import os
import numpy as np
import pandas as pd
import joblib


class CirrhosisAgent:
    """
    Agent de classification de la cirrhose.

    Le modèle est sauvegardé sous forme de package contenant :
      - model
      - feature_names
      - categorical_columns
      - numerical_columns
      - encoders
      - target_encoder
    """

    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Cirrhosis model not found: {model_path}"
            )

        self.model_path = model_path

        # ---------------------------------------------------------------------
        # Chargement du package
        # ---------------------------------------------------------------------
        package = joblib.load(model_path)

        if not isinstance(package, dict):
            raise TypeError(
                "Invalid cirrhosis model: expected a dictionary package."
            )

        # ---------------------------------------------------------------------
        # Récupération du vrai modèle
        # ---------------------------------------------------------------------
        self.model = package.get("model")

        if self.model is None:
            raise ValueError(
                "Invalid cirrhosis model: key 'model' is missing."
            )

        if not hasattr(self.model, "predict"):
            raise TypeError(
                "Invalid cirrhosis model: package['model'] "
                "does not provide a predict() method."
            )

        # ---------------------------------------------------------------------
        # Métadonnées du modèle
        # ---------------------------------------------------------------------
        self.feature_names = package.get("feature_names", [])

        self.categorical_columns = package.get(
            "categorical_columns", []
        )

        self.numerical_columns = package.get(
            "numerical_columns", []
        )

        self.encoders = package.get(
            "encoders", {}
        )

        self.target_encoder = package.get(
            "target_encoder", None
        )

        print("✅ CirrhosisAgent initialized")
        print(f"   Model       : {type(self.model).__name__}")
        print(f"   Features    : {len(self.feature_names)}")
        print(f"   Categorical : {len(self.categorical_columns)}")
        print(f"   Numerical   : {len(self.numerical_columns)}")

    # =========================================================================
    # PREPROCESSING
    # =========================================================================

    def _prepare_dataframe(self, data):
        """
        Prépare les données exactement selon les métadonnées
        enregistrées avec le modèle.
        """

        # DataFrame
        if isinstance(data, pd.DataFrame):
            df = data.copy()

        # Dictionnaire
        elif isinstance(data, dict):
            df = pd.DataFrame([data])

        # Liste / array
        else:
            arr = np.asarray(data)

            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            if self.feature_names and arr.shape[1] == len(self.feature_names):
                df = pd.DataFrame(
                    arr,
                    columns=self.feature_names
                )
            else:
                df = pd.DataFrame(arr)

        # ---------------------------------------------------------------------
        # Supprimer éventuellement la target si elle est fournie
        # ---------------------------------------------------------------------
        if "Stage" in df.columns:
            df = df.drop(columns=["Stage"])

        # ---------------------------------------------------------------------
        # Ajouter les colonnes manquantes
        # ---------------------------------------------------------------------
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan

        # ---------------------------------------------------------------------
        # Garder exactement les features du modèle
        # ---------------------------------------------------------------------
        if self.feature_names:
            df = df[self.feature_names]

        # ---------------------------------------------------------------------
        # Encodage des variables catégorielles
        # ---------------------------------------------------------------------
        for col in self.categorical_columns:

            if col not in df.columns:
                continue

            encoder = self.encoders.get(col)

            if encoder is None:
                continue

            # Convertir en chaîne pour correspondre au training
            values = df[col].astype(str)

            try:
                df[col] = encoder.transform(values)

            except ValueError:
                # Gestion des catégories inconnues
                classes = list(encoder.classes_)

                mapping = {
                    str(v): i
                    for i, v in enumerate(classes)
                }

                # Catégorie inconnue -> -1
                df[col] = values.map(mapping).fillna(-1).astype(float)

        # ---------------------------------------------------------------------
        # Conversion numérique
        # ---------------------------------------------------------------------
        for col in self.numerical_columns:

            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        # ---------------------------------------------------------------------
        # Remplacement des NaN
        #
        # Le modèle XGBoost peut généralement gérer les NaN directement.
        # On conserve donc les valeurs manquantes.
        # ---------------------------------------------------------------------

        return df

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict(self, data):
        """
        Effectue une prédiction.

        Retourne un dictionnaire standardisé utilisable
        par le coordinator.
        """

        df = self._prepare_dataframe(data)

        # ---------------------------------------------------------------------
        # Prediction
        # ---------------------------------------------------------------------
        prediction_encoded = self.model.predict(df)

        prediction_encoded = np.asarray(
            prediction_encoded
        ).reshape(-1)

        prediction = prediction_encoded[0]

        # ---------------------------------------------------------------------
        # Probabilités
        # ---------------------------------------------------------------------
        probability = None

        if hasattr(self.model, "predict_proba"):

            probabilities = self.model.predict_proba(df)

            probabilities = np.asarray(
                probabilities
            )

            if probabilities.ndim == 2:
                probability = float(
                    np.max(probabilities[0])
                )

        # ---------------------------------------------------------------------
        # Décodage de la classe
        # ---------------------------------------------------------------------
        predicted_label = prediction

        if self.target_encoder is not None:

            try:
                predicted_label = self.target_encoder.inverse_transform(
                    [int(prediction)]
                )[0]

            except Exception:
                predicted_label = prediction

        # ---------------------------------------------------------------------
        # Confiance
        # ---------------------------------------------------------------------
        confidence = probability if probability is not None else 0.0

        return {
            "prediction": predicted_label,
            "predicted_label": predicted_label,
            "probability": probability,
            "confidence": confidence,
            "agent": "cirrhosis",
            "status": "success"
        }

    # =========================================================================
    # ALIAS
    # =========================================================================

    def run(self, data):
        """Alias compatible avec l'orchestrateur."""
        return self.predict(data)
'''

agent_file.write_text(code, encoding="utf-8")

print("=" * 80)
print("✅ agents/cirrhosis_agent.py REMPLACÉ")
print("=" * 80)
