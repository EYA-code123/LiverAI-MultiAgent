# =============================================================================
# LiverAI-MultiAgent
# BASE AGENT
# =============================================================================

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAgent(ABC):
    """
    Base class for all LiverAI agents.

    Every specialized agent should return a standardized dictionary
    compatible with the LiverAI coordination layer.
    """

    def __init__(self, agent_id: str, model=None):
        self.agent_id = agent_id
        self.model = model

    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------

    @abstractmethod
    def predict(self, X):
        """
        Execute the agent prediction.

        Must return a dictionary containing at least:
            prediction
            probability
            confidence
            uncertainty
            quality
            details
            error
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # STANDARD RESULT
    # -------------------------------------------------------------------------

    def build_result(
        self,
        prediction: Any = None,
        probability: Any = None,
        confidence: float = 0.0,
        uncertainty: float = 1.0,
        quality: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        task_type: Optional[str] = None,
        status: Optional[str] = None,
        latency_ms: float = 0.0,
        missing_data_ratio: float = 0.0,
        explanation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a standardized agent result.

        This format is used by:
            AgentResult
            TrustManager
            ConflictDetector
            AdaptiveFusion
            DecisionEngine
            LiverAIOrchestrator
        """

        confidence = self._clip(confidence)
        uncertainty = self._clip(uncertainty)
        quality = self._clip(quality)
        missing_data_ratio = self._clip(missing_data_ratio)

        if status is None:
            status = "error" if error else "success"

        result = {
            "agent_id": self.agent_id,
            "agent": self.agent_id,

            "task_type": task_type,

            "prediction": prediction,
            "probability": probability,

            "confidence": confidence,
            "uncertainty": uncertainty,
            "quality": quality,

            "latency_ms": float(max(0.0, latency_ms)),
            "missing_data_ratio": missing_data_ratio,

            "details": details or {},

            "explanation": explanation,

            "status": status,
            "error": error,
        }

        return result

    # -------------------------------------------------------------------------
    # UTILS
    # -------------------------------------------------------------------------

    @staticmethod
    def _clip(value: Any) -> float:
        """
        Convert value to [0, 1].
        """

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0

        return max(0.0, min(1.0, value))
