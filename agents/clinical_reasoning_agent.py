# =============================================================================
# CLINICAL REASONING AGENT
# =============================================================================

import os
import time

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

    CPU-safe loading for checkpoints originally saved on CUDA.
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

    # =========================================================================
    # INIT
    # =========================================================================

    def __init__(self, model_package):

        print("=" * 70)
        print("CLINICAL REASONING AGENT")
        print("=" * 70)

        # ---------------------------------------------------------------------
        # MODEL PATH
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # DEVICE
        # ---------------------------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Loading Clinical Reasoning model from:")
        print(self.model_path)
        print("Device:", self.device)

        # ---------------------------------------------------------------------
        # CPU-SAFE MODEL LOADING
        # ---------------------------------------------------------------------

        original_torch_load = torch.load

        def cpu_safe_torch_load(*args, **kwargs):

            kwargs["map_location"] = torch.device("cpu")

            try:

                return original_torch_load(
                    *args,
                    **kwargs
                )

            except TypeError:

                kwargs.pop(
                    "weights_only",
                    None
                )

                return original_torch_load(
                    *args,
                    **kwargs
                )

        try:

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

            torch.load = original_torch_load

        # ---------------------------------------------------------------------
        # MODEL READY
        # ---------------------------------------------------------------------

        print(
            "✓ Clinical Reasoning model loaded successfully"
        )

        self.features = self.FEATURES.copy()

        self.target = self.TARGET

        print(
            "Features:",
            len(self.features)
        )

        print(
            "Target  :",
            self.target
        )

    # =========================================================================
    # DATAFRAME CREATION
    # =========================================================================

    def _create_dataframe(self, patient_data):

        # ---------------------------------------------------------------------
        # Convert input to DataFrame
        # ---------------------------------------------------------------------

        if isinstance(
            patient_data,
            pd.DataFrame
        ):

            df = patient_data.copy()

        elif isinstance(
            patient_data,
            dict
        ):

            df = pd.DataFrame(
                [patient_data]
            )

        elif isinstance(
            patient_data,
            (list, tuple, np.ndarray)
        ):

            array = np.asarray(
                patient_data
            )

            if array.ndim == 1:

                array = array.reshape(
                    1,
                    -1
                )

            df = pd.DataFrame(
                array,
                columns=self.features
            )

        else:

            raise TypeError(
                "Clinical input must be a dict, DataFrame, "
                "list, tuple or numpy array."
            )

        # ---------------------------------------------------------------------
        # Check required features
        # ---------------------------------------------------------------------

        missing_features = [
            feature
            for feature in self.features
            if feature not in df.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing Clinical Reasoning features: "
                + ", ".join(
                    missing_features
                )
            )

        # ---------------------------------------------------------------------
        # Keep ONLY expected features
        # ---------------------------------------------------------------------

        df = df[
            self.features
        ].copy()

        # ---------------------------------------------------------------------
        # Convert all features to numeric
        # ---------------------------------------------------------------------

        for feature in self.features:

            df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce"
            )

        # ---------------------------------------------------------------------
        # Check invalid values
        # ---------------------------------------------------------------------

        if df.isnull().any().any():

            missing = df.columns[
                df.isnull().any()
            ].tolist()

            raise ValueError(
                "Invalid or missing values in Clinical Reasoning "
                f"input: {missing}"
            )

        # ---------------------------------------------------------------------
        # IMPORTANT FIX
        #
        # PyTorch Tabular may internally assign floating-point
        # transformed values to these columns.
        #
        # If the original columns are int64, recent pandas versions
        # generate:
        #
        # FutureWarning:
        # Setting an item of incompatible dtype...
        #
        # Force every continuous input feature to float64.
        # ---------------------------------------------------------------------

        df = df.astype(
            {
                feature: "float64"
                for feature in self.features
            }
        )

        # ---------------------------------------------------------------------
        # Final clean DataFrame
        # ---------------------------------------------------------------------

        df = pd.DataFrame(
            df,
            columns=self.features
        )

        return df

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict(self, patient_data):

        start_time = time.time()

        try:

            # -----------------------------------------------------------------
            # Prepare input
            # -----------------------------------------------------------------

            df = self._create_dataframe(
                patient_data
            )

            # -----------------------------------------------------------------
            # Prediction
            # -----------------------------------------------------------------

            result = self.model.predict(
                df
            )

        except Exception as e:

            raise RuntimeError(
                "Clinical Reasoning prediction failed:\n"
                f"{e}"
            ) from e

        # ---------------------------------------------------------------------
        # Convert result to DataFrame
        # ---------------------------------------------------------------------

        if isinstance(
            result,
            pd.DataFrame
        ):

            prediction_df = result

        else:

            prediction_df = pd.DataFrame(
                result
            )

        if prediction_df.empty:

            raise RuntimeError(
                "Clinical Reasoning model returned "
                "an empty result."
            )

        # ---------------------------------------------------------------------
        # Find prediction column
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # Fallback search
        # ---------------------------------------------------------------------

        if prediction_column is None:

            for column in prediction_df.columns:

                name = str(
                    column
                ).lower()

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

        # ---------------------------------------------------------------------
        # Prediction value
        # ---------------------------------------------------------------------

        prediction = prediction_df[
            prediction_column
        ].iloc[0]

        if isinstance(
            prediction,
            np.generic
        ):

            prediction = prediction.item()

        try:

            if float(
                prediction
            ).is_integer():

                prediction = int(
                    prediction
                )

        except Exception:

            pass

        # ---------------------------------------------------------------------
        # Probabilities
        # ---------------------------------------------------------------------

        probabilities = {}

        for column in prediction_df.columns:

            name = str(
                column
            )

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

                    value = float(
                        value
                    )

                except Exception:

                    continue

                probabilities[name] = value

        # ---------------------------------------------------------------------
        # Confidence
        # ---------------------------------------------------------------------

        confidence = None

        if probabilities:

            confidence = max(
                probabilities.values()
            )

        # ---------------------------------------------------------------------
        # Output
        # ---------------------------------------------------------------------

        output = {

            "status": "success",

            "agent": self.AGENT_NAME,

            "model": self.MODEL_NAME,

            "prediction": prediction,

            "probabilities": probabilities,

            "quality": 1.0,

            "missing_ratio": 0.0,

            "features_used": self.features.copy(),

            "inference_time":
                time.time() - start_time,
        }

        if confidence is not None:

            output["confidence"] = confidence

            output["uncertainty"] = (
                1.0 - confidence
            )

        else:

            output["confidence"] = None

            output["uncertainty"] = None

        return output

    # =========================================================================
    # ANALYZE
    # =========================================================================

    def analyze(self, patient_data):

        return self.predict(
            patient_data
        )

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self):

        return {

            "agent": self.AGENT_NAME,

            "model": self.MODEL_NAME,

            "model_path": self.model_path,

            "device": str(
                self.device
            ),

            "loaded":
                self.model is not None,

            "status": "healthy",
        }
