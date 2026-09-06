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

    latency_ms: float = 0.0

    missing_data_ratio: float = 0.0

    trust: Optional[float] = None

    status: str = "success"

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    explanation: Optional[str] = None

    error: Optional[str] = None

    def to_dict(self):

        return {

            "agent_id":
                self.agent_id,

            "agent":
                self.agent_id,

            "task_type":
                self.task_type,

            "prediction":
                self.prediction,

            "probability":
                self.probability,

            "confidence":
                float(
                    self.confidence
                ),

            "uncertainty":
                float(
                    self.uncertainty
                ),

            "quality":
                float(
                    self.quality
                ),

            "latency_ms":
                float(
                    self.latency_ms
                ),

            "missing_data_ratio":
                float(
                    self.missing_data_ratio
                ),

            "trust":
                (
                    float(self.trust)
                    if self.trust is not None
                    else None
                ),

            "status":
                self.status,

            "details":
                self.details,

            "explanation":
                self.explanation,

            "error":
                self.error,
        }

    @classmethod
    def from_dict(
        cls,
        data
    ):

        if data is None:
            data = {}

        agent_id = data.get(
            "agent_id",
            data.get(
                "agent",
                "unknown"
            )
        )

        confidence = float(
            data.get(
                "confidence",
                0.0
            )
        )

        return cls(

            agent_id=str(
                agent_id
            ),

            task_type=str(
                data.get(
                    "task_type",
                    "unknown"
                )
            ),

            prediction=data.get(
                "prediction"
            ),

            probability=data.get(
                "probability"
            ),

            confidence=max(
                0.0,
                min(
                    1.0,
                    confidence
                )
            ),

            uncertainty=max(
                0.0,
                min(
                    1.0,
                    float(
                        data.get(
                            "uncertainty",
                            1.0 -
                            confidence
                        )
                    )
                )
            ),

            quality=max(
                0.0,
                min(
                    1.0,
                    float(
                        data.get(
                            "quality",
                            1.0
                        )
                    )
                )
            ),

            latency_ms=float(
                data.get(
                    "latency_ms",
                    0.0
                )
            ),

            missing_data_ratio=float(
                data.get(
                    "missing_data_ratio",
                    0.0
                )
            ),

            trust=(
                float(
                    data["trust"]
                )
                if data.get(
                    "trust"
                ) is not None
                else None
            ),

            status=data.get(
                "status",
                "success"
            ),

            details=data.get(
                "details",
                {}
            ),

            explanation=data.get(
                "explanation"
            ),

            error=data.get(
                "error"
            ),
        )
