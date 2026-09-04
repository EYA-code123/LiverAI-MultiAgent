import os
import pandas as pd
import numpy as np


class ClinicalReasoningAgent:

    def __init__(self, model_package=None):

        self.name = "ClinicalReasoningAgent"

        self.model_package = model_package
        self.model = None

        self.model_path = None

        if isinstance(model_package, dict):

            self.model = model_package.get("model")

            self.model_path = model_package.get(
                "model_path"
            )

        elif model_package is not None:

            self.model = model_package

        print(
            f"[{self.name}] initialized"
        )

    # ==========================================================
    # LOAD MODEL
    # ==========================================================

    @classmethod
    def from_pytorch_tabular(cls, model_path):

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"Clinical Reasoning model not found: "
                f"{model_path}"
            )

        from pytorch_tabular import TabularModel

        print(
            "[ClinicalReasoningAgent] "
            "Loading TabTransformer..."
        )

        model = TabularModel.load_model(
            model_path
        )

        print(
            "[ClinicalReasoningAgent] "
            "TabTransformer loaded successfully."
        )

        return cls(
            model_package={
                "model": model,
                "model_path": model_path
            }
        )

    # ==========================================================
    # PREPARE INPUT
    # ==========================================================

    def prepare_input(self, patient_data):

        required_features = [
            "mcv",
            "alkphos",
            "sgpt",
            "sgot",
            "gammagt",
            "drinks"
        ]

        if isinstance(patient_data, dict):

            data = patient_data.copy()

        elif isinstance(patient_data, pd.DataFrame):

            if len(patient_data) == 0:

                raise ValueError(
                    "patient_data DataFrame is empty."
                )

            data = patient_data.iloc[
                0
            ].to_dict()

        else:

            raise TypeError(
                "patient_data must be a "
                "dict or pandas DataFrame."
            )

        missing = [
            feature
            for feature in required_features
            if feature not in data
        ]

        if missing:

            raise ValueError(
                "Missing Clinical Reasoning "
                f"features: {missing}"
            )

        row = {
            feature: float(data[feature])
            for feature in required_features
        }

        return pd.DataFrame(
            [row],
            columns=required_features
        )

    # ==========================================================
    # PREDICT
    # ==========================================================

    def predict(self, patient_data):

        if self.model is None:

            raise RuntimeError(
                "Clinical Reasoning model is not loaded."
            )

        df = self.prepare_input(
            patient_data
        )

        prediction = self.model.predict(
            df
        )

        prediction_columns = [
            c
            for c in prediction.columns
            if "prediction" in c.lower()
        ]

        if not prediction_columns:

            raise RuntimeError(
                "No prediction column returned "
                "by TabTransformer."
            )

        prediction_column = (
            prediction_columns[0]
        )

        predicted_class = int(
            prediction[
                prediction_column
            ].iloc[0]
        )

        probability_columns = [
            c
            for c in prediction.columns
            if "probability" in c.lower()
        ]

        probabilities = {}

        for column in probability_columns:

            try:

                probabilities[column] = float(
                    prediction[
                        column
                    ].iloc[0]
                )

            except Exception:

                pass

        if predicted_class == 1:

            interpretation = (
                "Clinical profile classified "
                "in class 1."
            )

        else:

            interpretation = (
                "Clinical profile classified "
                "in class 0."
            )

        return {

            "agent": self.name,

            "prediction": predicted_class,

            "class": predicted_class,

            "probabilities": probabilities,

            "interpretation": interpretation,

            "features": df.iloc[0].to_dict()

        }

    # ==========================================================
    # HEALTH CHECK
    # ==========================================================

    def is_ready(self):

        return self.model is not None

    # ==========================================================
    # STRING REPRESENTATION
    # ==========================================================

    def __repr__(self):

        return (
            f"{self.__class__.__name__}("
            f"model_loaded={self.model is not None})"
        )
