from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentMessage:

    patient_id: Any

    agent_id: str

    prediction: Any

    probability: Any

    confidence: float

    uncertainty: float

    quality: float

    details: Dict[str, Any] = field(default_factory=dict)

    error: Optional[str] = None

    def to_dict(self):
        return {
            "patient_id": self.patient_id,
            "agent_id": self.agent_id,
            "prediction": self.prediction,
            "probability": self.probability,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "quality": self.quality,
            "details": self.details,
            "error": self.error
        }
