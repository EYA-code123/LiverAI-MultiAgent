# ============================================================
# CLINICAL REASONING AGENT
# CPU-SAFE VERSION
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import torch

from pytorch_tabular import TabularModel


class ClinicalReasoningAgent:
    """
    Clinical Reasoning Agent
    ------------------------
    Model:
        PyTorch Tabular / TabTransformer

    Features:
        mcv
        alkphos
        sgpt
        sgot
        gammagt
        drinks

    This implementation is CPU-safe and can load a checkpoint
    originally saved on CUDA.
    """

    AGENT_NAME = "ClinicalReasoningAgent"
    MODEL_NAME = "TabTransformer"

    FEATURES = [
        "mcv",
        "alkphos",
        "sgpt",
        "sgot",
        "gammagt",
        "drinks",
    ]

    TARGET = "selector"

    def __init__(self, model_package):

        print("=" * 70)
        print("CLINICAL REASONING AGENT")
        print("=" * 70)

        # --------------------------------------------------------
        # MODEL PATH
        # --------------------------------------------------------

        if not isinstance(model_package, str):
            raise TypeError(
                "ClinicalReasoningAgent expects the path to "
                "the PyTorch Tabular model directory."
            )

        self.model_path = model_package

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Clinical Reasoning model not found:\n"
                f"{self.model_path}"
            )

        # --------------------------------------------------------
        # DEVICE
        # --------------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Loading Clinical Reasoning model from:")
        print(self.model_path)
        print("Device:", self.device)

        # --------------------------------------------------------
        # CPU-SAFE MODEL LOADING
        # --------------------------------------------------------

        original_torch_load = torch.load

        def cpu_safe_torch_load(*args, **kwargs):

            # Force every serialized tensor to CPU.
            kwargs["map_location"] = torch.device("cpu")

            # PyTorch versions >= 2.6 may default to weights_only=True.
            # PyTorch Tabular checkpoints may require the complete
            # serialized object, so preserve the loader behavior when
            # possible.
            try:
                return original_torch_load(*args, **kwargs)
            except TypeError:
                kwargs.pop("weights_only", None)
                return original_torch_load(*args, **kwargs)

        try:

            # Patch torch.load only while PyTorch Tabular loads
            # the serialized model.
            torch.load = cpu_safe_torch_load

            self.model = TabularModel.load_model(
                self.model_path
            )

        except Exception as e:

            raise RuntimeError(
                "\nClinical Reasoning model could not be loaded.\n"
                "\n"
                "The model was probably serialized with CUDA.\n"
                "The loader attempted to force the checkpoint to CPU "
                "but the model could not be reconstructed.\n"
                "\n"
                f"Original error:\n{e}"
            ) from e

        finally:

            # ALWAYS restore the original torch.load.
            torch.load = original_torch_load

        # --------------------------------------------------------
        # MODEL READY
        # --------------------------------------------------------

        print("✓ Clinical Reasoning model loaded successfully")

        self.features = self.FEATURES.copy()
        self.target = self.TARGET

        print("Features:", len(self.features))
        print("Target  :", self.target)

    # ============================================================
    # DATAFRAME CREATION
    # ============================================================

    def _create_dataframe(self, patient_data):

        if isinstance(patient_data, pd.DataFrame):

            df = patient_data.copy()

        elif isinstance(patient_data, dict):

            df = pd.DataFrame([patient_data])

        elif isinstance(patient_data, (list, tuple, np.ndarray)):

            array = np.asarray(patient_data)

            if array.ndim == 1:
                array = array.reshape(1, -1)

            df = pd.DataFrame(
                array,
                columns=self.features
            )

        else:

            raise TypeError(
                "Clinical input must be a dict, DataFrame, "
                "list, tuple or numpy array."
            )

        # --------------------------------------------------------
        # CHECK FEATURES
        # --------------------------------------------------------

        missing_features = [
            feature
            for feature in self.features
            if feature not in df.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing Clinical Reasoning features: "
                + ", ".join(missing_features)
            )

        # Keep only expected features.
        df = df[self.features].copy()

        # Convert everything to numeric.
        for feature in self.features:

            df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce"
            )

        # Check NaN.
        if df.isnull().any().any():

            missing = df.columns[
                df.isnull().any()
            ].tolist()

            raise ValueError(
                "Invalid or missing values in Clinical Reasoning "
                f"input: {missing}"
            )

        return df

    # ============================================================
    # PREDICTION
    # ============================================================

    def predict(self, patient_data):

        df = self._create_dataframe(patient_data)

        try:

            result = self.model.predict(df)

        except Exception as e:

            raise RuntimeError(
                "Clinical Reasoning prediction failed:\n"
                f"{e}"
            ) from e

        # --------------------------------------------------------
        # CONVERT RESULT
        # --------------------------------------------------------

        if isinstance(result, pd.DataFrame):

            prediction_df = result

        else:

            prediction_df = pd.DataFrame(result)

        if prediction_df.empty:

            raise RuntimeError(
                "Clinical Reasoning model returned an empty result."
            )

        # --------------------------------------------------------
        # FIND PREDICTION COLUMN
        # --------------------------------------------------------

        prediction_column = None

        possible_prediction_columns = [
            "prediction",
            "Prediction",
            self.target,
            "selector_prediction",
        ]

        for column in possible_prediction_columns:

            if column in prediction_df.columns:
                prediction_column = column
                break

        # Fallback: search for columns containing prediction.
        if prediction_column is None:

            for column in prediction_df.columns:

                name = str(column).lower()

                if (
                    "prediction" in name
                    or name == self.target.lower()
                ):
                    prediction_column = column
                    break

        if prediction_column is None:

            raise RuntimeError(
                "Could not identify the prediction column.\n"
                f"Returned columns: "
                f"{list(prediction_df.columns)}"
            )

        prediction = prediction_df[
            prediction_column
        ].iloc[0]

        # Convert numpy scalar.
        if isinstance(prediction, np.generic):
            prediction = prediction.item()

        # Try integer conversion when appropriate.
        try:
            if float(prediction).is_integer():
                prediction = int(prediction)
        except Exception:
            pass

        # --------------------------------------------------------
        # PROBABILITIES
        # --------------------------------------------------------

        probabilities = {}

        for column in prediction_df.columns:

            name = str(column)

            lower_name = name.lower()

            if (
                "probability" in lower_name
                or "prob_" in lower_name
                or "prob" in lower_name
            ):

                value = prediction_df[
                    column
                ].iloc[0]

                try:
                    value = float(value)
                except Exception:
                    continue

                probabilities[name] = value

        # --------------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------------

        confidence = None

        if probabilities:

            confidence = max(
                probabilities.values()
            )

        # --------------------------------------------------------
        # OUTPUT
        # --------------------------------------------------------

        output = {
            "agent": self.AGENT_NAME,
            "model": self.MODEL_NAME,
            "prediction": prediction,
            "probabilities": probabilities,
            "status": "success",
        }

        if confidence is not None:

            output["confidence"] = confidence
            output["uncertainty"] = 1.0 - confidence

        return output

    # ============================================================
    # ANALYZE
    # ============================================================

    def analyze(self, patient_data):

        return self.predict(patient_data)

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def health_check(self):

        return {
            "agent": self.AGENT_NAME,
            "model": self.MODEL_NAME,
            "model_path": self.model_path,
            "device": str(self.device),
            "loaded": self.model is not None,
            "status": "healthy",
        }
