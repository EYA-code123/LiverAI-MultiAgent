# =============================================================================
# LIVERAI - CIRRHOSIS AGENT
# =============================================================================

import time
import numpy as np
import pandas as pd


class CirrhosisAgent:

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, model_path):

        self.model_path = model_path

        self.model = None
        self.target_encoder = None

        # Features expected by the trained XGBoost model
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
    # MODEL LOADING
    # =========================================================================

    def _load_model(self):

        try:

            import joblib

            self.model = joblib.load(
                self.model_path
            )

            print(
                f"✅ CirrhosisAgent model loaded: "
                f"{self.model_path}"
            )

            print(
                "   Model type:",
                type(self.model).__name__
            )

            print(
                "   Classes:",
                getattr(
                    self.model,
                    "classes_",
                    None
                )
            )

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
        # Dictionary input
        # ---------------------------------------------------------------------

        if isinstance(data, dict):

            df = pd.DataFrame(
                [data]
            )

        # ---------------------------------------------------------------------
        # DataFrame input
        # ---------------------------------------------------------------------

        elif isinstance(data, pd.DataFrame):

            df = data.copy()

        # ---------------------------------------------------------------------
        # Other tabular input
        # ---------------------------------------------------------------------

        else:

            try:

                df = pd.DataFrame(
                    data
                )

            except Exception as e:

                raise ValueError(
                    "Unsupported cirrhosis input type: "
                    f"{type(e).__name__}: {e}"
                )

        # ---------------------------------------------------------------------
        # Use model feature names whenever available
        # ---------------------------------------------------------------------

        model_features = getattr(
            self.model,
            "feature_names_in_",
            None
        )

        if model_features is not None:

            expected_features = [
                str(feature)
                for feature in model_features
            ]

        else:

            expected_features = self.features

        # ---------------------------------------------------------------------
        # Add missing columns
        # ---------------------------------------------------------------------

        for feature in expected_features:

            if feature not in df.columns:

                df[feature] = np.nan

        # ---------------------------------------------------------------------
        # Keep only expected features and exact order
        # ---------------------------------------------------------------------

        df = df[
            expected_features
        ].copy()

        # ---------------------------------------------------------------------
        # Convert values to numeric
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

            # -----------------------------------------------------------------
            # Prepare input
            # -----------------------------------------------------------------

            df = self._prepare_dataframe(
                data
            )

            # -----------------------------------------------------------------
            # Prediction
            # -----------------------------------------------------------------

            raw_prediction = self.model.predict(
                df
            )

            raw_prediction = np.asarray(
                raw_prediction
            ).reshape(-1)

            if len(raw_prediction) == 0:

                raise ValueError(
                    "The cirrhosis model returned "
                    "an empty prediction."
                )

            prediction = raw_prediction[0]

            # -----------------------------------------------------------------
            # Validate prediction against model classes
            # -----------------------------------------------------------------

            model_classes = getattr(
                self.model,
                "classes_",
                None
            )

            if model_classes is not None:

                valid_classes = np.asarray(
                    model_classes
                ).reshape(-1)

                if not any(
                    prediction == cls
                    for cls in valid_classes
                ):

                    raise ValueError(
                        f"Invalid model prediction "
                        f"{prediction}. "
                        f"Expected one of "
                        f"{valid_classes.tolist()}."
                    )

            # -----------------------------------------------------------------
            # Probability
            # -----------------------------------------------------------------

            probability = None
            class_probabilities = None

            if hasattr(
                self.model,
                "predict_proba"
            ):

                try:

                    probabilities = (
                        self.model.predict_proba(
                            df
                        )
                    )

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
                            np.max(
                                probabilities[0]
                            )
                        )

                except Exception:

                    probability = None
                    class_probabilities = None

            # -----------------------------------------------------------------
            # IMPORTANT:
            #
            # Do NOT apply target_encoder.inverse_transform().
            #
            # The current XGBoost model already has:
            #
            # classes_ = [0, 1, 2]
            #
            # The previous target encoder transformed class 2 into "3.0",
            # which produced an invalid prediction.
            # -----------------------------------------------------------------

            predicted_label = prediction

            # -----------------------------------------------------------------
            # Convert NumPy scalar
            # -----------------------------------------------------------------

            if isinstance(
                predicted_label,
                np.generic
            ):

                predicted_label = (
                    predicted_label.item()
                )

            # -----------------------------------------------------------------
            # Keep integer class labels as integers
            # -----------------------------------------------------------------

            if isinstance(
                predicted_label,
                (float, np.floating)
            ):

                if float(
                    predicted_label
                ).is_integer():

                    predicted_label = int(
                        predicted_label
                    )

            # -----------------------------------------------------------------
            # Confidence
            # -----------------------------------------------------------------

            confidence = (
                probability
                if probability is not None
                else 0.0
            )

            # -----------------------------------------------------------------
            # Uncertainty
            # -----------------------------------------------------------------

            uncertainty = max(
                0.0,
                min(
                    1.0,
                    1.0 - confidence
                )
            )

            # -----------------------------------------------------------------
            # Missing data
            # -----------------------------------------------------------------

            missing_data_ratio = float(
                df.isna()
                .mean()
                .mean()
            )

            # -----------------------------------------------------------------
            # Quality
            # -----------------------------------------------------------------

            quality = max(
                0.0,
                min(
                    1.0,
                    1.0 - missing_data_ratio
                )
            )

            # -----------------------------------------------------------------
            # Latency
            # -----------------------------------------------------------------

            latency_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            # -----------------------------------------------------------------
            # Final result
            # -----------------------------------------------------------------

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
                    predicted_label,

                "predicted_label":
                    predicted_label,

                "probability":
                    probability,

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
        # ERROR HANDLING
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

        return self.predict(
            data
        )
