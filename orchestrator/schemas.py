# =============================================================================
# LiverAI-MultiAgent
# STANDARD SCHEMAS
# =============================================================================

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentResult:
    """
    Standard representation of an agent prediction.

    All agents must communicate through this structure.
    """

    agent_id: str

    prediction: Any = None

    probability: Any = None

    confidence: float = 0.0

    uncertainty: float = 1.0

    quality: float = 0.0

    trust: float = 0.5

    status: str = "success"

    task_type: str = "unknown"

    latency_ms: float = 0.0

    missing_data_ratio: float = 0.0

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    error: Optional[str] = None

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

        self.trust = self._clip(
            self.trust
        )

        self.missing_data_ratio = self._clip(
            self.missing_data_ratio
        )

        if self.error is not None:
            self.status = "error"

    @staticmethod
    def _clip(value):

        try:
            value = float(value)
        except Exception:
            value = 0.0

        return max(
            0.0,
            min(1.0, value)
        )

    @property
    def success(self):

        return (
            self.status == "success"
            and self.error is None
            and self.prediction is not None
        )

    def to_dict(self):

        return {
            "agent_id": self.agent_id,
            "prediction": self.prediction,
            "probability": self.probability,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "quality": self.quality,
            "trust": self.trust,
            "status": self.status,
            "task_type": self.task_type,
            "latency_ms": self.latency_ms,
            "missing_data_ratio":
                self.missing_data_ratio,
            "details": self.details,
            "error": self.error
        }
