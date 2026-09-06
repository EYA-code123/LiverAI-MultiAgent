# =============================================================================
# LIVERAI - CIRRHOSIS AGENT
# =============================================================================

import time
import numpy as np
import pandas as pd
import joblib


class CirrhosisAgent:

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, model_path):

        self.model_path = model_path

        self.model = None
        self.target_encoder = None

        self.features = [
            "N_Days",
            "Status",
            "Drug",
            "Age",
            "Sex",
            "Ascites",
            "Hepatomegaly",
            "Spiders",
            "Edema",
            "Bilirubin",
            "Cholesterol",
            "Albumin",
            "Copper",
            "Alk_Phos",
            "SGOT",
            "Tryglicerides",
            "Platelets",
            "Prothrombin"
        ]

        self._load_model()

    # =========================================================================
    # LOAD MODEL
    # =========================================================================

    def _load_model(self):

        try:

            self.model = joblib.load(
                self.model_path
            )

            print("=" * 70)
            print("CIRRHOSIS AGENT INITIALIZED")
            print("=" * 70)
            print("Model path :", self.model_path)
            print(
                "Model type :",
                type(self.model).__name__
            )

            model_features = getattr(
                self.model,
                "feature_names_in_",
                None
            )

            if model_features is not None:

                self.features = [
                    str(x)
                    for x in model_features
                ]

            print(
                "Features   :",
                len(self.features)
            )

            print(
                "Classes    :",
                getattr(
                    self.model,
                    "classes_",
                    None
                )
            )

            print("=" * 70)

        except Exception as e:

            raise RuntimeError(
                "Unable to load Cirrhosis model: "
                f"{type(e).__name__}: {e}"
            )

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================

    def _prepare_dataframe(self, data):

        # ---------------------------------------------------------------------
        # Dictionary
        # ---------------------------------------------------------------------

        if isinstance(data, dict):

            df = pd.DataFrame([data])

        # ---------------------------------------------------------------------
        # DataFrame
        # ---------------------------------------------------------------------

        elif isinstance(data, pd.DataFrame):

            df = data.copy()

        # ---------------------------------------------------------------------
        # Other tabular input
        # ---------------------------------------------------------------------

        else:

            try:

                df = pd.DataFrame(data)

            except Exception as e:

                raise ValueError(
                    "Unsupported input type: "
                    f"{type(e).__name__}: {e}"
                )

        # ---------------------------------------------------------------------
        # Add missing features
        # ---------------------------------------------------------------------

        for feature in self.features:

            if feature not in df.columns:

                df[feature] = np.nan

        # ---------------------------------------------------------------------
        # Exact feature order
        # ---------------------------------------------------------------------

        df = df[
            self.features
        ].copy()

        # ---------------------------------------------------------------------
        # Numeric conversion
        # ---------------------------------------------------------------------

        for column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        return df

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict(self, data):

        start_time = time.perf_counter()

        try:

            # =================================================================
            # PREPARE DATA
            # =================================================================

            df = self._prepare_dataframe(
                data
            )

            # =================================================================
            # MODEL CLASSES
            # =================================================================

            model_classes = getattr(
                self.model,
                "classes_",
                None
            )

            if model_classes is None:

                raise ValueError(
                    "The cirrhosis model does not expose "
                    "`classes_`."
                )

            model_classes = np.asarray(
                model_classes
            ).reshape(-1)

            # =================================================================
            # PROBABILITY-BASED PREDICTION
            # =================================================================
            #
            # IMPORTANT:
            #
            # The current XGBoost model produces:
            #
            # model.classes_ = [0, 1, 2]
            #
            # but model.predict() returns 3.0.
            #
            # Therefore we use predict_proba() and map the winning
            # probability back to model.classes_.
            #
            # Example:
            #
            # probabilities = [0.26, 0.22, 0.51]
            # argmax          = 2
            # classes_[2]     = 2
            #
            # Final prediction = 2
            #
            # =================================================================

            probabilities = None
            class_probabilities = None

            if hasattr(
                self.model,
                "predict_proba"
            ):

                probabilities = (
                    self.model.predict_proba(
                        df
                    )
                )

                probabilities = np.asarray(
                    probabilities
                )

                if probabilities.ndim != 2:

                    raise ValueError(
                        "Unexpected probability output shape: "
                        f"{probabilities.shape}"
                    )

                if probabilities.shape[0] == 0:

                    raise ValueError(
                        "The model returned no probability."
                    )

                if probabilities.shape[1] != len(
                    model_classes
                ):

                    raise ValueError(
                        "Number of probability columns "
                        "does not match model classes. "
                        f"probabilities={probabilities.shape}, "
                        f"classes={model_classes.shape}"
                    )

                # First sample
                sample_probabilities = (
                    probabilities[0]
                    .astype(float)
                )

                # Store probabilities
                class_probabilities = (
                    sample_probabilities
                    .tolist()
                )

                # Winning probability index
                predicted_index = int(
                    np.argmax(
                        sample_probabilities
                    )
                )

                # Map index -> actual model class
                prediction = model_classes[
                    predicted_index
                ]

                probability = float(
                    sample_probabilities[
                        predicted_index
                    ]
                )

            else:

                # =============================================================
                # FALLBACK
                # =============================================================

                raw_prediction = (
                    self.model.predict(df)
                )

                raw_prediction = np.asarray(
                    raw_prediction
                ).reshape(-1)

                if len(raw_prediction) == 0:

                    raise ValueError(
                        "The model returned an empty prediction."
                    )

                prediction = raw_prediction[0]

                probability = None

            # =================================================================
            # VALIDATE PREDICTION
            # =================================================================

            if not any(
                prediction == cls
                for cls in model_classes
            ):

                raise ValueError(
                    f"Invalid prediction {prediction}. "
                    f"Expected one of "
                    f"{model_classes.tolist()}."
                )

            # =================================================================
            # CONVERT NUMPY VALUE
            # =================================================================

            if isinstance(
                prediction,
                np.generic
            ):

                prediction = prediction.item()

            # =================================================================
            # FORCE INTEGER FOR INTEGER CLASS LABELS
            # =================================================================

            if isinstance(
                prediction,
                (float, np.floating)
            ):

                if float(
                    prediction
                ).is_integer():

                    prediction = int(
                        prediction
                    )

            # =================================================================
            # CONFIDENCE
            # =================================================================

            confidence = (
                float(probability)
                if probability is not None
                else 0.0
            )

            # =================================================================
            # UNCERTAINTY
            # =================================================================

            uncertainty = max(
                0.0,
                min(
                    1.0,
                    1.0 - confidence
                )
            )

            # =================================================================
            # MISSING DATA
            # =================================================================

            missing_data_ratio = float(
                df.isna()
                .mean()
                .mean()
            )

            # =================================================================
            # QUALITY
            # =================================================================

            quality = max(
                0.0,
                min(
                    1.0,
                    1.0 - missing_data_ratio
                )
            )

            # =================================================================
            # LATENCY
            # =================================================================

            latency_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            # =================================================================
            # RESULT
            # =================================================================

            return {

                "agent":
                    "CirrhosisAgent",

                "task_type":
                    "cirrhosis_classification",

                "model":
                    type(
                        self.model
                    ).__name__,

                "prediction":
                    prediction,

                "predicted_label":
                    prediction,

                "probability":
                    confidence,

                "confidence":
                    confidence,

                "uncertainty":
                    uncertainty,

                "quality":
                    quality,

                "trust":
                    None,

                "class_probabilities":
                    class_probabilities,

                "model_classes":
                    model_classes.tolist(),

                "features_used":
                    list(
                        df.columns
                    ),

                "missing_data_ratio":
                    missing_data_ratio,

                "latency_ms":
                    float(
                        latency_ms
                    ),

                "status":
                    "success",

                "error":
                    None
            }

        # =====================================================================
        # ERROR
        # =====================================================================

        except Exception as e:

            latency_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            return {

                "agent":
                    "CirrhosisAgent",

                "task_type":
                    "cirrhosis_classification",

                "model":
                    type(
                        self.model
                    ).__name__
                    if self.model is not None
                    else None,

                "prediction":
                    None,

                "predicted_label":
                    None,

                "probability":
                    None,

                "confidence":
                    0.0,

                "uncertainty":
                    1.0,

                "quality":
                    0.0,

                "trust":
                    None,

                "class_probabilities":
                    None,

                "model_classes":
                    getattr(
                        self.model,
                        "classes_",
                        None
                    ).tolist()
                    if getattr(
                        self.model,
                        "classes_",
                        None
                    ) is not None
                    else None,

                "features_used":
                    [],

                "missing_data_ratio":
                    None,

                "latency_ms":
                    float(
                        latency_ms
                    ),

                "status":
                    "error",

                "error":
                    f"{type(e).__name__}: {e}"
            }

    # =========================================================================
    # ALIAS
    # =========================================================================

    def analyze(self, data):

        return self.predict(data)
