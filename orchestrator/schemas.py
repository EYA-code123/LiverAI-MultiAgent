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

    This object is shared by the coordination layer.
    """

    agent_id: str

    task_type: str = "unknown"

    prediction: Any = None
    probability: Any = None

    confidence: float = 0.0
    uncertainty: float = 1.0
    quality: float = 0.0

    latency_ms: float = 0.0
    missing_data_ratio: float = 0.0

    trust: Optional[float] = None

    status: str = "success"

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    explanation: Optional[str] = None

    error: Optional[str] = None

    # -------------------------------------------------------------------------
    # CONVERSION
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent": self.agent_id,

            "task_type": self.task_type,

            "prediction": self.prediction,
            "probability": self.probability,

            "confidence": float(self.confidence),
            "uncertainty": float(self.uncertainty),
            "quality": float(self.quality),

            "latency_ms": float(self.latency_ms),
            "missing_data_ratio": float(
                self.missing_data_ratio
            ),

            "trust": (
                float(self.trust)
                if self.trust is not None
                else None
            ),

            "status": self.status,

            "details": self.details,

            "explanation": self.explanation,

            "error": self.error,
        }

    # -------------------------------------------------------------------------
    # FACTORY
    # -------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Convert a dictionary returned by an agent into AgentResult.
        """

        if data is None:
            data = {}

        agent_id = data.get(
            "agent_id",
            data.get("agent", "unknown")
        )

        details = data.get("details") or {}

        task_type = data.get(
            "task_type",
            details.get("task_type", "unknown")
        )

        status = data.get("status")

        if status is None:
            status = (
                "error"
                if data.get("error")
                else "success"
            )

        confidence = data.get(
            "confidence",
            data.get("probability", 0.0)
            if isinstance(
                data.get("probability"),
                (int, float)
            )
            else 0.0
        )

        uncertainty = data.get(
            "uncertainty",
            1.0 - float(confidence)
        )

        quality = data.get(
            "quality",
            1.0 if not data.get("error") else 0.0
        )

        return cls(
            agent_id=str(agent_id),

            task_type=str(task_type),

            prediction=data.get("prediction"),

            probability=data.get("probability"),

            confidence=float(
                max(0.0, min(1.0, float(confidence)))
            ),

            uncertainty=float(
                max(0.0, min(1.0, float(uncertainty)))
            ),

            quality=float(
                max(0.0, min(1.0, float(quality)))
            ),

            latency_ms=float(
                data.get("latency_ms", 0.0)
            ),

            missing_data_ratio=float(
                max(
                    0.0,
                    min(
                        1.0,
                        float(
                            data.get(
                                "missing_data_ratio",
                                0.0
                            )
                        )
                    )
                )
            ),

            trust=(
                float(data["trust"])
                if data.get("trust") is not None
                else None
            ),

            status=status,

            details=details,

            explanation=data.get(
                "explanation"
            ),

            error=data.get("error"),
        )
