# communication/message.py

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime


@dataclass
class AgentMessage:

    patient_id: Any

    agent_id: str

    model_version: str

    task_type: str

    prediction: Any

    probabilities: Any

    confidence: float

    uncertainty: float

    data_quality: float

    missing_data_ratio: float

    feature_importance: Dict[str, float] = field(
        default_factory=dict
    )

    embedding: Any = None

    explanation: Optional[str] = None

    latency_ms: float = 0.0

    reliability: float = 0.5

    utility: float = 0.5

    stability: float = 0.5

    historical_performance: float = 0.5

    trust: float = 0.5

    modality_available: bool = True

    timestamp: str = field(
        default_factory=lambda:
        datetime.utcnow().isoformat()
    )

    status: str = "success"

    error: Optional[str] = None

    def to_dict(self):

        return {

            "patient_id":
                self.patient_id,

            "agent_id":
                self.agent_id,

            "model_version":
                self.model_version,

            "task_type":
                self.task_type,

            "prediction":
                self.prediction,

            "probabilities":
                self.probabilities,

            "confidence":
                self.confidence,

            "uncertainty":
                self.uncertainty,

            "data_quality":
                self.data_quality,

            "missing_data_ratio":
                self.missing_data_ratio,

            "feature_importance":
                self.feature_importance,

            "embedding":
                self.embedding,

            "explanation":
                self.explanation,

            "latency_ms":
                self.latency_ms,

            "reliability":
                self.reliability,

            "utility":
                self.utility,

            "stability":
                self.stability,

            "historical_performance":
                self.historical_performance,

            "trust":
                self.trust,

            "modality_available":
                self.modality_available,

            "timestamp":
                self.timestamp,

            "status":
                self.status,

            "error":
                self.error
        }
