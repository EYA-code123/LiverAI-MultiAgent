from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentResult:

    agent_id: str

    prediction: Any = None

    probability: Any = None

    confidence: float = 0.0

    uncertainty: float = 1.0

    quality: float = 0.0

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    error: Optional[str] = None

    def to_dict(self):

        return {
            "agent_id": self.agent_id,
            "prediction": self.prediction,
            "probability": self.probability,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "quality": self.quality,
            "details": self.details,
            "error": self.error
        }
