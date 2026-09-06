from typing import Any, Dict
import numpy as np


class AgentAdapter:
    """
    Converts heterogeneous specialist-agent outputs
    into the unified LiverAI message format.
    """

    REQUIRED_FIELDS = [
        "agent_id",
        "model_version",
        "task_type",
        "prediction",
        "probabilities",
        "confidence",
        "uncertainty",
        "data_quality",
        "missing_data_ratio",
        "feature_importance",
        "embedding",
        "explanation",
        "latency_ms",
        "status",
        "error",
    ]

    @staticmethod
    def _clip01(value, default=0.0):
        try:
            return float(np.clip(float(value), 0.0, 1.0))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _extract_probabilities(result):
        probabilities = result.get("probabilities")

        if probabilities is not None:
            return probabilities

        probabilities = result.get("class_probabilities")

        if probabilities is not None:
            return probabilities

        probability = result.get("probability")

        if probability is not None:
            return probability

        return None

    @staticmethod
    def normalize(agent_id: str, result: Dict[str, Any]):
        if result is None:
            return {
                "agent_id": agent_id,
                "model_version": "unknown",
                "task_type": "unknown",
                "prediction": None,
                "probabilities": None,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "data_quality": 0.0,
                "missing_data_ratio": 1.0,
                "feature_importance": {},
                "embedding": None,
                "explanation": None,
                "latency_ms": 0.0,
                "status": "unavailable",
                "error": "No result returned by agent",
            }

        if not isinstance(result, dict):
            result = {
                "prediction": result,
                "status": "success",
            }

        status = result.get("status", "success")

        prediction = result.get(
            "prediction",
            result.get("predicted_label")
        )

        probabilities = AgentAdapter._extract_probabilities(result)

        confidence = result.get("confidence")

        if confidence is None:
            confidence = result.get("probability")

        confidence = AgentAdapter._clip01(
            confidence,
            default=0.0
        )

        uncertainty = result.get("uncertainty")

        if uncertainty is None:
            uncertainty = 1.0 - confidence

        uncertainty = AgentAdapter._clip01(
            uncertainty,
            default=1.0 - confidence
        )

        quality = result.get("data_quality")

        if quality is None:
            quality = result.get("quality")

        missing_ratio = result.get("missing_data_ratio")

        if missing_ratio is None:
            missing_ratio = result.get("missing_ratio")

        if missing_ratio is None:
            missing_ratio = 1.0 - AgentAdapter._clip01(
                quality,
                default=0.0
            )

        missing_ratio = AgentAdapter._clip01(
            missing_ratio,
            default=0.0
        )

        if quality is None:
            quality = 1.0 - missing_ratio

        quality = AgentAdapter._clip01(
            quality,
            default=1.0 - missing_ratio
        )

        feature_importance = result.get(
            "feature_importance",
            {}
        )

        embedding = result.get(
            "embedding"
        )

        explanation = result.get(
            "explanation"
        )

        latency = result.get("latency_ms")

        if latency is None:
            latency = result.get("inference_time", 0.0)

        try:
            latency = float(latency)

            # Some agents report inference_time in seconds.
            if "inference_time" in result and "latency_ms" not in result:
                latency *= 1000.0

        except (TypeError, ValueError):
            latency = 0.0

        task_type = result.get(
            "task_type",
            "unknown"
        )

        model_version = result.get(
            "model_version",
            result.get("model", "unknown")
        )

        return {
            "agent_id": str(
                result.get(
                    "agent_id",
                    result.get(
                        "agent",
                        agent_id
                    )
                )
            ),

            "model_version": str(
                model_version
            ),

            "task_type": str(
                task_type
            ),

            "prediction": prediction,

            "probabilities": probabilities,

            "confidence": confidence,

            "uncertainty": uncertainty,

            "data_quality": quality,

            "missing_data_ratio": missing_ratio,

            "feature_importance": (
                feature_importance
                if isinstance(
                    feature_importance,
                    dict
                )
                else {}
            ),

            "embedding": embedding,

            "explanation": explanation,

            "latency_ms": latency,

            "status": status,

            "error": result.get("error"),
        }

    @classmethod
    def validate(cls, result):
        missing = [
            field
            for field in cls.REQUIRED_FIELDS
            if field not in result
        ]

        return {
            "valid": len(missing) == 0,
            "missing_fields": missing,
        }
