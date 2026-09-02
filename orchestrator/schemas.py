from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentResult:

    agent_id: str

    task_type: str = "unknown"

    prediction: Any = None
    probability: Any = None

    confidence: float = 0.0
    uncertainty: float = 1.0

    quality: float = 0.0
    missing_data_ratio: float = 0.0

    latency_ms: float = 0.0

    trust: float = 0.5

    explanation: Optional[str] = None

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    error: Optional[str] = None

    @property
    def status(self):

        if self.error is not None:
            return "error"

        return "success"

    def to_dict(self):

        return {
            "agent_id": self.agent_id,
            "task_type": self.task_type,

            "prediction": self.prediction,
            "probability": self.probability,

            "confidence": self.confidence,
            "uncertainty": self.uncertainty,

            "quality": self.quality,
            "missing_data_ratio":
                self.missing_data_ratio,

            "latency_ms":
                self.latency_ms,

            "trust":
                self.trust,

            "explanation":
                self.explanation,

            "details":
                self.details,

            "error":
                self.error,

            "status":
                self.status
        }
