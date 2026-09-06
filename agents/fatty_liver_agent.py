# =============================================================================
# FATTY LIVER AGENT
# =============================================================================

import time
import warnings
import numpy as np
import pandas as pd


class FattyLiverAgent:
    """
    Agent de classification du fatty liver.

    Modèle :
        LightGBM Pipeline

    Entrées :
        mcv
        alkphos
        sgpt
        sgot
        gammagt
        drinks

    Sortie :
        prediction
        confidence
        uncertainty
        class_probabilities
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, model_package):

        self.model_name = "LightGBM Pipeline"

        self.feature_names = [
            "mcv",
            "alkphos",
            "sgpt",
            "sgot",
            "gammagt",
            "drinks"
        ]

        self.target_name = "selector"

        self.model = None
        self.target_classes = None

        # ---------------------------------------------------------------------
        # Load model
        # ---------------------------------------------------------------------

        if isinstance(model_package, dict):

            if "model" not in model_package:
                raise ValueError(
                    "model_package doit contenir une clé 'model'."
                )

            self.model = model_package["model"]

            if "target_classes" in model_package:
                self.target_classes = model_package["target_classes"]

            elif "classes" in model_package:
                self.target_classes = model_package["classes"]

        else:
            # Le modèle actuel est directement un sklearn Pipeline
            self.model = model_package

        if self.model is None:
            raise ValueError(
                "Le modèle FattyLiver n'a pas pu être chargé."
            )

        if not hasattr(self.model, "predict"):
            raise TypeError(
                "Le modèle fourni doit posséder une méthode predict()."
            )

        # ---------------------------------------------------------------------
        # Try to recover classes automatically
        # ---------------------------------------------------------------------

        if self.target_classes is None:

            if hasattr(self.model, "classes_"):
                self.target_classes = list(self.model.classes_)

            elif hasattr(self.model, "named_steps"):

                classifier = self.model.named_steps.get("classifier")

                if classifier is not None and hasattr(
                    classifier,
                    "classes_"
                ):
                    self.target_classes = list(classifier.classes_)

        # Convert classes to strings for stable JSON output
        if self.target_classes is not None:
            self.target_classes = [
                str(value) for value in self.target_classes
            ]

    # =========================================================================
    # INPUT PREPARATION
    # =========================================================================

    def _prepare_input(self, patient_data):

        # ---------------------------------------------------------------------
        # Convert input to DataFrame
        # ---------------------------------------------------------------------

        if isinstance(patient_data, pd.DataFrame):

            X = patient_data.copy()

        elif isinstance(patient_data, dict):

            X = pd.DataFrame([patient_data])

        else:

            raise TypeError(
                "patient_data doit être un dictionnaire "
                "ou un pandas.DataFrame."
            )

        # ---------------------------------------------------------------------
        # Remove target if accidentally provided
        # ---------------------------------------------------------------------

        if self.target_name in X.columns:

            X = X.drop(
                columns=[self.target_name]
            )

        # ---------------------------------------------------------------------
        # Add missing features
        # ---------------------------------------------------------------------

        for feature in self.feature_names:

            if feature not in X.columns:

                X[feature] = np.nan

        # ---------------------------------------------------------------------
        # Keep only expected features
        # ---------------------------------------------------------------------

        X = X[self.feature_names].copy()

        # ---------------------------------------------------------------------
        # Convert features to numeric
        # ---------------------------------------------------------------------

        for feature in self.feature_names:

            X[feature] = pd.to_numeric(
                X[feature],
                errors="coerce"
            )

        return X

    # =========================================================================
    # SAFE MODEL PREDICTION
    # =========================================================================

    def _predict_safely(self, X):

        """
        Exécute predict() en supprimant uniquement le warning LightGBM
        connu concernant les feature names.

        Le modèle lui-même n'est PAS modifié.
        """

        with warnings.catch_warnings():

            warnings.filterwarnings(
                "ignore",
                message=(
                    "X does not have valid feature names, "
                    "but LGBMClassifier was fitted with feature names"
                ),
                category=UserWarning
            )

            prediction = self.model.predict(X)

        return prediction

    # =========================================================================
    # SAFE PROBABILITY PREDICTION
    # =========================================================================

    def _predict_proba_safely(self, X):

        """
        Exécute predict_proba() en supprimant uniquement le warning
        LightGBM lié aux feature names.
        """

        with warnings.catch_warnings():

            warnings.filterwarnings(
                "ignore",
                message=(
                    "X does not have valid feature names, "
                    "but LGBMClassifier was fitted with feature names"
                ),
                category=UserWarning
            )

            probabilities = self.model.predict_proba(X)

        return probabilities

    # =========================================================================
    # PREDICT
    # =========================================================================

    def predict(self, patient_data):

        start_time = time.time()

        try:

            # -----------------------------------------------------------------
            # Prepare input
            # -----------------------------------------------------------------

            X = self._prepare_input(patient_data)

            # -----------------------------------------------------------------
            # Missing data
            # -----------------------------------------------------------------

            missing_ratio = float(
                X.isna().sum().sum() / X.size
            )

            quality = max(
                0.0,
                1.0 - missing_ratio
            )

            # -----------------------------------------------------------------
            # Prediction
            # -----------------------------------------------------------------

            prediction = self._predict_safely(X)

            prediction_value = prediction[0]

            # -----------------------------------------------------------------
            # Probability
            # -----------------------------------------------------------------

            probabilities = None
            confidence = 0.0
            uncertainty = 1.0
            class_probabilities = {}

            if hasattr(
                self.model,
                "predict_proba"
            ):

                probabilities = np.asarray(
                    self._predict_proba_safely(X)[0],
                    dtype=float
                )

                confidence = float(
                    np.max(probabilities)
                )

                uncertainty = float(
                    1.0 - confidence
                )

                # -------------------------------------------------------------
                # Recover target classes if necessary
                # -------------------------------------------------------------

                if self.target_classes is None:

                    if hasattr(
                        self.model,
                        "classes_"
                    ):

                        self.target_classes = [
                            str(value)
                            for value in self.model.classes_
                        ]

                    elif hasattr(
                        self.model,
                        "named_steps"
                    ):

                        classifier = (
                            self.model
                            .named_steps
                            .get("classifier")
                        )

                        if (
                            classifier is not None
                            and hasattr(
                                classifier,
                                "classes_"
                            )
                        ):

                            self.target_classes = [
                                str(value)
                                for value in classifier.classes_
                            ]

                # -------------------------------------------------------------
                # Build probability dictionary
                # -------------------------------------------------------------

                if self.target_classes is None:

                    self.target_classes = [
                        str(i)
                        for i in range(
                            len(probabilities)
                        )
                    ]

                for i, probability in enumerate(
                    probabilities
                ):

                    if i < len(
                        self.target_classes
                    ):

                        class_name = (
                            self.target_classes[i]
                        )

                    else:

                        class_name = str(i)

                    class_probabilities[
                        class_name
                    ] = float(probability)

            # -----------------------------------------------------------------
            # Inference time
            # -----------------------------------------------------------------

            inference_time = (
                time.time() - start_time
            )

            # -----------------------------------------------------------------
            # Result
            # -----------------------------------------------------------------

            return {

                "status": "success",

                "agent": "FattyLiverAgent",

                "model": self.model_name,

                "prediction": str(
                    prediction_value
                ),

                "confidence": confidence,

                "uncertainty": uncertainty,

                "quality": quality,

                "missing_ratio": missing_ratio,

                "class_probabilities":
                    class_probabilities,

                "features_used":
                    self.feature_names,

                "inference_time":
                    inference_time
            }

        except Exception as e:

            inference_time = (
                time.time() - start_time
            )

            return {

                "status": "error",

                "agent": "FattyLiverAgent",

                "model": self.model_name,

                "prediction": None,

                "confidence": 0.0,

                "uncertainty": 1.0,

                "quality": 0.0,

                "missing_ratio": 1.0,

                "class_probabilities": {},

                "features_used":
                    self.feature_names,

                "inference_time":
                    inference_time,

                "error": str(e)
            }
