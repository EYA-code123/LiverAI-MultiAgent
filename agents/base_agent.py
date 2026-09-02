from abc import ABC, abstractmethod


class BaseAgent(ABC):

    def __init__(
        self,
        agent_id,
        model=None,
        task_type="unknown"
    ):
        self.agent_id = agent_id
        self.model = model
        self.task_type = task_type

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def build_result(
        self,
        prediction=None,
        probability=None,
        confidence=0.0,
        uncertainty=1.0,
        quality=0.0,
        missing_data_ratio=0.0,
        latency_ms=0.0,
        feature_importance=None,
        embedding=None,
        explanation=None,
        details=None,
        error=None
    ):

        return {
            "agent_id": self.agent_id,
            "task_type": self.task_type,

            "prediction": prediction,
            "probability": probability,

            "confidence": float(
                max(0.0, min(1.0, confidence))
            ),

            "uncertainty": float(
                max(0.0, min(1.0, uncertainty))
            ),

            "quality": float(
                max(0.0, min(1.0, quality))
            ),

            "missing_data_ratio": float(
                max(
                    0.0,
                    min(1.0, missing_data_ratio)
                )
            ),

            "latency_ms": float(
                max(0.0, latency_ms)
            ),

            "feature_importance":
                feature_importance,

            "embedding":
                embedding,

            "explanation":
                explanation,

            "details":
                details or {},

            "error":
                error
        }
