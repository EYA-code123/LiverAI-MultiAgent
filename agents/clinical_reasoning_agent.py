from pathlib import Path

clinical_file = Path(
    "/content/LiverAI-MultiAgent/agents/clinical_reasoning_agent.py"
)

clinical_code = r'''
import os
import numpy as np
import pandas as pd
import torch

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
        # DEVICE
        # ============================================================

        # Force CPU because the current Colab runtime has no CUDA.
        self.device = torch.device("cpu")

        # ============================================================
        # VALIDATION DU CHEMIN
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
                "ClinicalReasoningAgent: model directory not found:\n"
                f"{self.model_path}"
            )

        # ============================================================
        # CHARGEMENT TABTRANSFORMER
        # ============================================================

        print(
            "Loading Clinical Reasoning model from:"
        )
        print(self.model_path)
        print("Device: CPU")

        try:
            # Pytorch Tabular uses PyTorch internally.
            # The model package was saved with CUDA tensors.
            #
            # First try the standard loader.
            self.model = TabularModel.load_model(
                self.model_path
            )

        except RuntimeError as e:

            error_message = str(e)

            if (
                "deserialize object on a CUDA device" in error_message
                or "torch.cuda.is_available() is False" in error_message
                or "CUDA" in error_message
            ):
                raise RuntimeError(
                    "\nClinical Reasoning model was saved with CUDA "
                    "and cannot currently be loaded in this CPU-only "
                    "runtime using the installed PyTorch Tabular loader.\n\n"
                    "The model checkpoint must be converted to CPU "
                    "or loaded with map_location='cpu'.\n\n"
                    f"Original error:\n{error_message}"
                ) from e

            raise

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

        print("=" * 70)
        print("CLINICAL REASONING AGENT")
        print("=" * 70)
        print("✓ Model loaded successfully")
        print(f"✓ Model       : {self.model_name}")
        print(f"✓ Device      : {self.device}")
        print(f"✓ Features    : {self.feature_names}")
        print("=" * 70)

    # ================================================================
    # CREATE DATAFRAME
    # ================================================================

    def _create_dataframe(self, patient_data):

        # ------------------------------------------------------------
        # DataFrame
        # ------------------------------------------------------------

        if isinstance(patient_data, pd.DataFrame):

            df = patient_data.copy()

        # ------------------------------------------------------------
        # Dictionary
        # ------------------------------------------------------------

        elif isinstance(patient_data, dict):

            df = pd.DataFrame([patient_data])

        # ------------------------------------------------------------
        # List / tuple / numpy array
        # ------------------------------------------------------------

        elif isinstance(
            patient_data,
            (list, tuple, np.ndarray)
        ):

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
        # VERIFY FEATURES
        # ============================================================

        missing = [
            col
            for col in self.FEATURE_COLUMNS
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                "Missing clinical features: "
                + ", ".join(missing)
            )

        # ============================================================
        # KEEP ONLY MODEL FEATURES
        # ============================================================

        df = df[self.FEATURE_COLUMNS].copy()

        # ============================================================
        # NUMERIC CONVERSION
        # ============================================================

        for col in self.FEATURE_COLUMNS:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # ============================================================
        # MISSING VALUES
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
        # Prepare data
        # ------------------------------------------------------------

        df = self._create_dataframe(patient_data)

        # ------------------------------------------------------------
        # Prediction
        # ------------------------------------------------------------

        prediction = self.model.predict(df)

        # ============================================================
        # FIND PREDICTION COLUMNS
        # ============================================================

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

        # ============================================================
        # PREDICTED CLASS
        # ============================================================

        if prediction_columns:

            predicted_class = int(
                prediction[prediction_columns[0]].iloc[0]
            )

        else:

            raise RuntimeError(
                "TabTransformer prediction column not found."
            )

        # ============================================================
        # PROBABILITIES
        # ============================================================

        probabilities = {}

        for col in probability_columns:

            try:

                value = float(
                    prediction[col].iloc[0]
                )

            except Exception:

                continue

            probabilities[col] = value

        # ============================================================
        # CONFIDENCE
        # ============================================================

        confidence = None

        if probabilities:

            confidence = max(
                probabilities.values()
            )

        # ============================================================
        # RESULT
        # ============================================================

        result = {
            "agent": self.name,
            "model": self.model_name,
            "prediction": predicted_class,
            "probabilities": probabilities,
            "confidence": confidence,
            "status": "success",
            "device": str(self.device),
            "quality": 1.0,
            "missing_data_ratio": 0.0,
            "modality": "clinical"
        }

        return result

    # ================================================================
    # ANALYZE
    # ================================================================

    def analyze(self, patient_data):

        return self.predict(patient_data)

    # ================================================================
    # CALL
    # ================================================================

    def __call__(self, patient_data):

        return self.predict(patient_data)
'''

clinical_file.write_text(
    clinical_code,
    encoding="utf-8"
)

print("=" * 70)
print("CLINICAL REASONING AGENT UPDATED")
print("=" * 70)
print(f"✓ File written: {clinical_file}")
