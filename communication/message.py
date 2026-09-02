from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import json
import time


@dataclass
class AgentMessage:
    """
    Standard communication format for every LiverAI specialist agent.

    This implements the unified message structure required by
    Phase 1 of the Adaptive Coordination Intelligence guide.
    """

    patient_id: Any
    agent_id: str

    # Core prediction
    prediction: Any = None
    probability: Any = None

    # Uncertainty
    confidence: float = 0.0
    uncertainty: float = 1.0

    # Data / model quality
    quality: float = 0.0
    missing_data_ratio: float = 0.0

    # Adaptive coordination
    trust: float = 0.5
    utility: float = 0.0
    agreement: float = 0.0
    stability: float = 0.0

    # Metadata
    task_type: str = "unknown"
    modality: str = "unknown"
    latency_ms: float = 0.0

    # Additional information
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    embeddings: Any = None

    details: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: str = "success"
    error: Optional[str] = None

    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        """Convert message to a JSON-compatible dictionary."""

        data = asdict(self)

        # Avoid serializing very large embeddings by default
        if data.get("embeddings") is not None:
            try:
                if hasattr(data["embeddings"], "tolist"):
                    data["embeddings"] = data["embeddings"].tolist()
            except Exception:
                data["embeddings"] = None

        return data

    def to_json(self):
        """Serialize message."""

        return json.dumps(
            self.to_dict(),
            default=str,
            indent=2
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create an AgentMessage from a dictionary."""

        allowed = {
            "patient_id",
            "agent_id",
            "prediction",
            "probability",
            "confidence",
            "uncertainty",
            "quality",
            "missing_data_ratio",
            "trust",
            "utility",
            "agreement",
            "stability",
            "task_type",
            "modality",
            "latency_ms",
            "class_probabilities",
            "feature_importance",
            "explanation",
            "embeddings",
            "details",
            "status",
            "error",
            "timestamp",
        }

        clean_data = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return cls(**clean_data)
