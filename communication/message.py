from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime


@dataclass
class AgentMessage:
    """
    Standard message exchanged between specialist agents
    and the Adaptive Coordination Intelligence layer.
    """

    patient_id: Any
    agent_id: str
    task_type: str

    prediction: Any = None
    probability: Any = None

    confidence: float = 0.0
    uncertainty: float = 1.0

    quality: float = 0.0
    missing_data_ratio: float = 0.0

    latency_ms: float = 0.0

    feature_importance: Optional[Dict[str, float]] = None
    embedding: Any = None

    explanation: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    error: Optional[str] = None

    timestamp: str = field(
        default_factory=lambda:
        datetime.utcnow().isoformat()
    )

    def __post_init__(self):

        self.confidence = self._clip(
            self.confidence
        )

        self.uncertainty = self._clip(
            self.uncertainty
        )

        self.quality = self._clip(
            self.quality
        )

        self.missing_data_ratio = self._clip(
            self.missing_data_ratio
        )

        self.latency_ms = max(
            0.0,
            float(self.latency_ms)
        )

    @staticmethod
    def _clip(value):

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0

        return max(
            0.0,
            min(1.0, value)
        )

    @property
    def status(self):

        if self.error is not None:
            return "error"

        return "success"

    def to_dict(self):

        return {
            "patient_id": self.patient_id,
            "agent_id": self.agent_id,
            "task_type": self.task_type,

            "prediction": self.prediction,
            "probability": self.probability,

            "confidence": self.confidence,
            "uncertainty": self.uncertainty,

            "quality": self.quality,
            "missing_data_ratio":
                self.missing_data_ratio,

            "latency_ms": self.latency_ms,

            "feature_importance":
                self.feature_importance,

            "embedding": self.embedding,

            "explanation": self.explanation,

            "metadata": self.metadata,

            "error": self.error,
            "timestamp": self.timestamp,

            "status": self.status
        }
