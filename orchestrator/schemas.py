from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentResult:

    agent: str

    status: str = "success"

    prediction: Any = None

    probability: Optional[float] = None

    probabilities: Optional[Dict[str, float]] = None

    confidence: Optional[float] = None

    uncertainty: Optional[float] = None

    quality: Optional[float] = None

    trust: Optional[float] = None

    model: Optional[str] = None

    modality: Optional[str] = None

    recommended_action: Optional[str] = None

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
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "quality": self.quality,
            "trust": self.trust,
            "model": self.model,
            "modality": self.modality,
            "recommended_action": self.recommended_action,
            "details": self.details,
            "error": self.error
        }
