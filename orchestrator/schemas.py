from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentResult:

    agent: str
    status: str

    prediction: Any = None
    probability: Optional[float] = None
    confidence: Optional[float] = None

    model: Optional[str] = None

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    error: Optional[str] = None

    def to_dict(self):

        return {
            "agent": self.agent,
            "status": self.status,
            "prediction": self.prediction,
            "probability": self.probability,
            "confidence": self.confidence,
            "model": self.model,
            "details": self.details,
            "error": self.error
        }
