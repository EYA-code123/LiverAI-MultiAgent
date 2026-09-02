# agents/base_agent.py

from abc import ABC, abstractmethod
import time
import numpy as np


class BaseAgent(ABC):

    def __init__(
        self,
        agent_id,
        model=None,
        model_version="1.0"
    ):
        self.agent_id = agent_id
        self.model = model
        self.model_version = model_version

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    # ---------------------------------------------------------
    # DATA QUALITY
    # ---------------------------------------------------------

    @staticmethod
    def compute_missing_ratio(data):

        if data is None:
            return 1.0

        try:
            if hasattr(data, "isnull"):
                total = data.size

                if total == 0:
                    return 1.0

                return float(data.isnull().sum().sum() / total)

            if isinstance(data, dict):

                if len(data) == 0:
                    return 1.0

                missing = sum(
                    value is None or (
                        isinstance(value, float)
                        and np.isnan(value)
                    )
                    for value in data.values()
                )

                return float(missing / len(data))

        except Exception:
            pass

        return 0.0

    @staticmethod
    def compute_data_quality(missing_ratio):

        return float(
            np.clip(
                1.0 - missing_ratio,
                0.0,
                1.0
            )
        )

    # ---------------------------------------------------------
    # UNCERTAINTY
    # ---------------------------------------------------------

    @staticmethod
    def probability_uncertainty(probabilities):

        if probabilities is None:
            return 1.0

        probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        if len(probabilities) == 0:
            return 1.0

        confidence = np.max(probabilities)

        return float(
            np.clip(
                1.0 - confidence,
                0.0,
                1.0
            )
        )

    # ---------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------

    def extract_feature_importance(self):

        if self.model is None:
            return {}

        try:

            if hasattr(
                self.model,
                "feature_importances_"
            ):

                values = self.model.feature_importances_

                names = getattr(
                    self,
                    "feature_names",
                    None
                )

                if names is not None:

                    return {
                        str(name): float(value)
                        for name, value
                        in zip(names, values)
                    }

        except Exception:
            pass

        return {}

    # ---------------------------------------------------------
    # EMBEDDING
    # ---------------------------------------------------------

    def extract_embedding(self, X):

        """
        Generic embedding extraction.

        Models that expose an internal representation
        can override this method.
        """

        return None

    # ---------------------------------------------------------
    # EXPLANATION
    # ---------------------------------------------------------

    def generate_explanation(
        self,
        prediction,
        confidence,
        feature_importance
    ):

        if not feature_importance:

            return (
                f"{self.agent_id} predicted "
                f"{prediction} with confidence "
                f"{confidence:.3f}."
            )

        top_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        features = ", ".join(
            f"{name}={value:.3f}"
            for name, value in top_features
        )

        return (
            f"{self.agent_id} predicted "
            f"{prediction} with confidence "
            f"{confidence:.3f}. "
            f"Most influential features: {features}."
        )

    # ---------------------------------------------------------
    # STANDARD RESULT
    # ---------------------------------------------------------

    def build_result(
        self,
        prediction=None,
        probabilities=None,
        confidence=0.0,
        uncertainty=1.0,
        data_quality=0.0,
        missing_data_ratio=1.0,
        feature_importance=None,
        embedding=None,
        explanation=None,
        latency_ms=0.0,
        status="success",
        error=None,
        task_type=None
    ):

        return {

            "agent_id":
                self.agent_id,

            "model_version":
                self.model_version,

            "task_type":
                task_type,

            "prediction":
                prediction,

            "probabilities":
                probabilities,

            "confidence":
                float(
                    np.clip(
                        confidence,
                        0.0,
                        1.0
                    )
                ),

            "uncertainty":
                float(
                    np.clip(
                        uncertainty,
                        0.0,
                        1.0
                    )
                ),

            "data_quality":
                float(
                    np.clip(
                        data_quality,
                        0.0,
                        1.0
                    )
                ),

            "missing_data_ratio":
                float(
                    np.clip(
                        missing_data_ratio,
                        0.0,
                        1.0
                    )
                ),

            "feature_importance":
                feature_importance or {},

            "embedding":
                embedding,

            "explanation":
                explanation,

            "latency_ms":
                float(latency_ms),

            "status":
                status,

            "error":
                error
        }

    # ---------------------------------------------------------
    # TIMED PREDICTION
    # ---------------------------------------------------------

    def start_timer(self):
        return time.perf_counter()

    @staticmethod
    def elapsed_ms(start):
        return (
            time.perf_counter() - start
        ) * 1000.0
