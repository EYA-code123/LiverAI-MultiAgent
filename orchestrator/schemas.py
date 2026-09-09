# =============================================================================
# LiverAI-MultiAgent
# ORCHESTRATOR SCHEMAS
# =============================================================================

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentResult:

    agent_id: str

    task_type: str = "unknown"

    modality: str = "unknown"

    prediction: Any = None

    probability: Any = None

    confidence: float = 0.0

    uncertainty: float = 1.0

    quality: float = 0.0

    latency_ms: float = 0.0

    missing_data_ratio: float = 0.0

    trust: Optional[float] = None

    agreement: Optional[float] = None

    status: str = "success"

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    explanation: Optional[str] = None

    error: Optional[str] = None

    # =========================================================================
    # TO DICT
    # =========================================================================

    def to_dict(self):

        return {

            "agent_id":
                self.agent_id,

            "agent":
                self.agent_id,

            "task_type":
                self.task_type,

            "modality":
                self.modality,

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
                    float(
                        self.trust
                    )
                    if self.trust is not None
                    else None
                ),

            "agreement":
                (
                    float(
                        self.agreement
                    )
                    if self.agreement is not None
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

    # =========================================================================
    # FROM DICT
    # =========================================================================

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

        confidence = cls._safe_float(
            data.get(
                "confidence",
                0.0
            )
        )

        uncertainty = cls._safe_float(
            data.get(
                "uncertainty",
                1.0 - confidence
            )
        )

        quality = cls._safe_float(
            data.get(
                "quality",
                1.0
            )
        )

        missing_ratio = cls._safe_float(
            data.get(
                "missing_data_ratio",
                0.0
            )
        )

        trust = data.get(
            "trust"
        )

        agreement = data.get(
            "agreement"
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

            modality=str(
                data.get(
                    "modality",
                    "unknown"
                )
            ),

            prediction=data.get(
                "prediction"
            ),

            probability=data.get(
                "probability"
            ),

            confidence=confidence,

            uncertainty=uncertainty,

            quality=quality,

            latency_ms=cls._safe_float(
                data.get(
                    "latency_ms",
                    0.0
                ),
                allow_above_one=True
            ),

            missing_data_ratio=
                missing_ratio,

            trust=(
                cls._safe_float(
                    trust
                )
                if trust is not None
                else None
            ),

            agreement=(
                cls._safe_float(
                    agreement
                )
                if agreement is not None
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

    # =========================================================================
    # SAFE FLOAT
    # =========================================================================

    @staticmethod
    def _safe_float(
        value,
        default=0.0,
        allow_above_one=False
    ):

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            value = default

        if allow_above_one:

            return max(
                0.0,
                value
            )

        return max(
            0.0,
            min(
                1.0,
                value
            )
        )


__all__ = [
    "AgentResult",
]
