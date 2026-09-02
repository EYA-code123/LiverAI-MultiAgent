# =============================================================================
# LiverAI-MultiAgent
# STANDARD AGENT MESSAGE
# =============================================================================

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentMessage:

    patient_id: Any

    agent_id: str

    prediction: Any = None

    probability: Any = None

    confidence: float = 0.0

    uncertainty: float = 1.0

    quality: float = 0.0

    trust: float = 0.5

    task_type: str = "unknown"

    latency_ms: float = 0.0

    missing_data_ratio: float = 0.0

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    error: Optional[str] = None

    status: str = "success"

    def to_dict(self):

        return {
            "patient_id": self.patient_id,
            "agent_id": self.agent_id,
            "prediction": self.prediction,
            "probability": self.probability,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "quality": self.quality,
            "trust": self.trust,
            "task_type": self.task_type,
            "latency_ms": self.latency_ms,
            "missing_data_ratio":
                self.missing_data_ratio,
            "details": self.details,
            "error": self.error,
            "status": self.status
        }
