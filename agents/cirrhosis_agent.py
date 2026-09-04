# =============================================================================
# CIRRHOSIS AGENT - VERSION ROBUSTE
# =============================================================================

import os
import time
import numpy as np
import pandas as pd
import joblib


class CirrhosisAgent:
    """
    Agent de classification de la cirrhose.

    Compatible avec :
        1. un package dict contenant :
           {
               "model": model,
               "feature_names": [...],
               "categorical_columns": [...],
               "numerical_columns": [...],
               "encoders": {...},
               "target_encoder": ...
           }

        2. un modèle directement sauvegardé.
    """

    def __init__(self, model_path):
        if not isinstance(model_path, (str, os.PathLike)):
            raise TypeError(
                "model_path doit être un chemin vers le modèle."
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Cirrhosis model not found:\n{model_path}"
            )

        self.model_path = str(model_path)

        # ---------------------------------------------------------------------
        # Chargement
        # ---------------------------------------------------------------------
        self._load_model()

        print("=" * 70)
        print("CIRRHOSIS AGENT INITIALIZED")
        print("=" * 70)
        print(f"Model path : {self.model_path}")
        print(f"Model type : {type(self.model).__name__}")
        print(f"Features   : {len(self.feature_names)}")
        print("=" * 70)

    # =========================================================================
    # LOAD MODEL
    # =========================================================================

    def _load_model(self):

        try:
            package = joblib.load(self.model_path)

        except Exception as e:
            raise RuntimeError(
                "\nImpossible de charger le modèle Cirrhosis.\n"
                f"Fichier : {self.model_path}\n"
                f"Erreur  : {type(e).__name__}: {e}\n\n"
                "Le problème vient probablement du fichier .pkl "
                "lui-même et non de CirrhosisAgent."
            ) from e

        # ---------------------------------------------------------------------
        # CAS 1 : package
        # ---------------------------------------------------------------------
        if isinstance(package, dict):

            self.model = package.get("model")

            if self.model is None:
                raise ValueError(
                    "Le package Cirrhosis ne contient pas la clé 'model'."
                )

            self.feature_names = package.get(
                "feature_names",
                []
            )

            self.categorical_columns = package.get(
                "categorical_columns",
                []
            )

            self.numerical_columns = package.get(
                "numerical_columns",
                []
            )

            self.encoders = package.get(
                "encoders",
                {}
            )

            self.target_encoder = package.get(
                "target_encoder",
                None
            )

        # ---------------------------------------------------------------------
        # CAS 2 : modèle directement sauvegardé
        # ---------------------------------------------------------------------
        else:

            self.model = package

            self.feature_names = []
            self.categorical_columns = []
            self.numerical_columns = []
            self.encoders = {}
            self.target_encoder = None

            # Essayer de récupérer les features depuis XGBoost
            try:

                if hasattr(self.model, "feature_names_in_"):

                    self.feature_names = list(
                        self.model.feature_names_in_
                    )

                elif hasattr(self.model, "get_booster"):

                    booster = self.model.get_booster()

                    if booster.feature_names is not None:
                        self.feature_names = list(
                            booster.feature_names
                        )

            except Exception:
                pass

        # ---------------------------------------------------------------------
        # Vérification
        # ---------------------------------------------------------------------
        if not hasattr(self.model, "predict"):
            raise TypeError(
                "L'objet chargé n'est pas un modèle compatible "
                "avec predict()."
            )

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================

    def _prepare_dataframe(self, data):

        # ---------------------------------------------------------------------
        # DataFrame
        # ---------------------------------------------------------------------
        if isinstance(data, pd.DataFrame):

            df = data.copy()

        # ---------------------------------------------------------------------
        # Dict
        # ---------------------------------------------------------------------
        elif isinstance(data, dict):

            df = pd.DataFrame([data])

        # ---------------------------------------------------------------------
        # Array / list
        # ---------------------------------------------------------------------
        else:

            arr = np.asarray(data)

            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            if self.feature_names:

                if arr.shape[1] != len(self.feature_names):

                    raise ValueError(
                        f"Nombre de features incorrect.\n"
                        f"Reçu       : {arr.shape[1]}\n"
                        f"Attendu    : {len(self.feature_names)}\n"
                        f"Features   : {self.feature_names}"
                    )

                df = pd.DataFrame(
                    arr,
                    columns=self.feature_names
                )

            else:

                df = pd.DataFrame(arr)

        # ---------------------------------------------------------------------
        # Supprimer target
        # ---------------------------------------------------------------------
        target_columns = [
            "Stage",
            "stage",
            "target",
            "Target"
        ]

        for col in target_columns:

            if col in df.columns:
                df = df.drop(columns=[col])

        # ---------------------------------------------------------------------
        # Vérification / ajout des features
        # ---------------------------------------------------------------------
        if self.feature_names:

            missing = [
                col
                for col in self.feature_names
                if col not in df.columns
            ]

            if missing:

                raise ValueError(
                    "Features Cirrhosis manquantes : "
                    + ", ".join(missing)
                )

            df = df[self.feature_names]

        # ---------------------------------------------------------------------
        # Encodage catégoriel
        # ---------------------------------------------------------------------
        for col in self.categorical_columns:

            if col not in df.columns:
                continue

            encoder = self.encoders.get(col)

            if encoder is None:
                continue

            values = df[col].astype(str)

            try:

                df[col] = encoder.transform(values)

            except Exception:

                classes = list(
                    getattr(
                        encoder,
                        "classes_",
                        []
                    )
                )

                mapping = {
                    str(value): index
                    for index, value in enumerate(classes)
                }

                df[col] = (
                    values
                    .map(mapping)
                    .fillna(-1)
                    .astype(float)
                )

        # ---------------------------------------------------------------------
        # Conversion numérique
        # ---------------------------------------------------------------------
        for col in self.numerical_columns:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        return df

    # =========================================================================
    # PREDICT
    # =========================================================================

    def predict(self, data):

        start_time = time.perf_counter()

        try:

            df = self._prepare_dataframe(data)

            # ---------------------------------------------------------------
            # Prediction
            # ---------------------------------------------------------------
            raw_prediction = self.model.predict(df)

            raw_prediction = np.asarray(
                raw_prediction
            ).reshape(-1)

            prediction = raw_prediction[0]

            # ---------------------------------------------------------------
            # Probability
            # ---------------------------------------------------------------
            probability = None
            class_probabilities = None

            if hasattr(self.model, "predict_proba"):

                try:

                    probabilities = self.model.predict_proba(df)

                    probabilities = np.asarray(
                        probabilities
                    )

                    if probabilities.ndim == 2:

                        class_probabilities = (
                            probabilities[0]
                            .astype(float)
                            .tolist()
                        )

                        probability = float(
                            np.max(probabilities[0])
                        )

                except Exception:
                    probability = None

            # ---------------------------------------------------------------
            # Target decoding
            # ---------------------------------------------------------------
            predicted_label = prediction

            if self.target_encoder is not None:

                try:

                    predicted_label = (
                        self.target_encoder
                        .inverse_transform(
                            [int(prediction)]
                        )[0]
                    )

                except Exception:
                    predicted_label = prediction

            # ---------------------------------------------------------------
            # Convert numpy values
            # ---------------------------------------------------------------
            if isinstance(
                predicted_label,
                np.generic
            ):
                predicted_label = predicted_label.item()

            # ---------------------------------------------------------------
            # Confidence
            # ---------------------------------------------------------------
            confidence = (
                probability
                if probability is not None
                else 0.0
            )

            # ---------------------------------------------------------------
            # Latency
            # ---------------------------------------------------------------
            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            return {

                "agent": "CirrhosisAgent",

                "task_type":
                    "cirrhosis_classification",

                "model":
                    type(self.model).__name__,

                "prediction":
                    predicted_label,

                "predicted_label":
                    predicted_label,

                "probability":
                    probability,

                "confidence":
                    confidence,

                "class_probabilities":
                    class_probabilities,

                "features_used":
                    list(df.columns),

                "missing_data_ratio":
                    float(
                        df.isna()
                        .mean()
                        .mean()
                    ),

                "latency_ms":
                    float(latency_ms),

                "status":
                    "success",

                "error":
                    None
            }

        except Exception as e:

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            return {

                "agent":
                    "CirrhosisAgent",

                "task_type":
                    "cirrhosis_classification",

                "prediction":
                    None,

                "predicted_label":
                    None,

                "probability":
                    None,

                "confidence":
                    0.0,

                "class_probabilities":
                    None,

                "features_used":
                    [],

                "missing_data_ratio":
                    None,

                "latency_ms":
                    float(latency_ms),

                "status":
                    "error",

                "error":
                    f"{type(e).__name__}: {e}"
            }

    # =========================================================================
    # RUN
    # =========================================================================

    def run(self, data):

        return self.predict(data)
