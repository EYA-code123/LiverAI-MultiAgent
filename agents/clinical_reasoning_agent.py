from pathlib import Path

ROOT = Path("/content/LiverAI-MultiAgent")
agent_file = ROOT / "agents" / "clinical_reasoning_agent.py"

code = r'''
import os
import numpy as np
import pandas as pd

from pytorch_tabular import TabularModel


class ClinicalReasoningAgent:
    """
    Clinical Reasoning Agent basé sur le TabTransformer
    entraîné avec PyTorch Tabular sur le dataset BUPA.
    """

    FEATURE_COLUMNS = [
        "mcv",
        "alkphos",
        "sgpt",
        "sgot",
        "gammagt",
        "drinks",
    ]

    def __init__(self, model_package):
        """
        Parameters
        ----------
        model_package : str
            Chemin vers le dossier contenant le modèle
            PyTorch Tabular sauvegardé.
        """

        if not isinstance(model_package, (str, os.PathLike)):
            raise TypeError(
                "ClinicalReasoningAgent: model_package doit être "
                "le chemin du dossier du modèle PyTorch Tabular."
            )

        self.model_path = os.fspath(model_package)

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"❌ Dossier du modèle introuvable : {self.model_path}"
            )

        required_files = [
            "config.yml",
            "datamodule.sav",
            "callbacks.sav",
            "model.ckpt",
            "custom_params.sav",
        ]

        missing = [
            filename
            for filename in required_files
            if not os.path.exists(
                os.path.join(self.model_path, filename)
            )
        ]

        if missing:
            raise FileNotFoundError(
                "❌ Fichiers du modèle manquants : "
                + ", ".join(missing)
            )

        print("Loading Clinical Reasoning TabTransformer...")

        self.model = TabularModel.load_model(self.model_path)

        print(f"✓ Clinical Reasoning model loaded")
        print(f"✓ Model path: {self.model_path}")

    def _prepare_input(self, patient_data):
        """
        Prépare les données d'entrée dans exactement le même
        format que pendant l'entraînement.
        """

        if isinstance(patient_data, pd.DataFrame):
            df = patient_data.copy()

        elif isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])

        elif isinstance(patient_data, (list, tuple, np.ndarray)):
            values = np.asarray(patient_data).reshape(-1)

            if len(values) != len(self.FEATURE_COLUMNS):
                raise ValueError(
                    f"❌ Nombre de variables incorrect : "
                    f"{len(values)} reçues, "
                    f"{len(self.FEATURE_COLUMNS)} attendues."
                )

            df = pd.DataFrame(
                [values],
                columns=self.FEATURE_COLUMNS
            )

        else:
            raise TypeError(
                "patient_data doit être un dictionnaire, "
                "DataFrame, liste, tuple ou numpy array."
            )

        missing = [
            col
            for col in self.FEATURE_COLUMNS
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                "❌ Variables cliniques manquantes : "
                + ", ".join(missing)
            )

        df = df[self.FEATURE_COLUMNS].copy()

        for col in self.FEATURE_COLUMNS:
            df[col] = pd.to_numeric(
                df[col],
                errors="raise"
            )

        return df

    def predict(self, patient_data):
        """
        Effectue une prédiction clinique.

        Returns
        -------
        dict
            Résultat contenant la classe prédite et les probabilités.
        """

        df = self._prepare_input(patient_data)

        prediction_df = self.model.predict(df)

        # Trouver automatiquement la colonne de prédiction
        prediction_columns = [
            col
            for col in prediction_df.columns
            if "prediction" in col.lower()
        ]

        if not prediction_columns:
            raise RuntimeError(
                "❌ Colonne de prédiction introuvable dans "
                "la sortie PyTorch Tabular."
            )

        prediction_column = prediction_columns[0]

        prediction = int(
            prediction_df[prediction_column].iloc[0]
        )

        result = {
            "prediction": prediction
        }

        # Récupération des probabilités
        probability_columns = [
            col
            for col in prediction_df.columns
            if "probability" in col.lower()
        ]

        for col in probability_columns:
            value = prediction_df[col].iloc[0]

            try:
                result[col] = float(value)
            except (TypeError, ValueError):
                pass

        # Informations lisibles
        if "selector_0_probability" in result:
            result["probability_class_0"] = result[
                "selector_0_probability"
            ]

        if "selector_1_probability" in result:
            result["probability_class_1"] = result[
                "selector_1_probability"
            ]

        result["model"] = "TabTransformer-BUPA"

        return result

    def analyze(self, patient_data):
        """
        Alias utilisé par l'orchestrateur.
        """
        return self.predict(patient_data)
'''

agent_file.parent.mkdir(parents=True, exist_ok=True)
agent_file.write_text(code.strip() + "\n", encoding="utf-8")

print("✓ clinical_reasoning_agent.py remplacé")
print(agent_file)
