from abc import ABC, abstractmethod


class BaseAgent(ABC):

    def __init__(self, agent_id, model=None):
        self.agent_id = agent_id
        self.model = model

    @abstractmethod
    def predict(self, X):
        """
        Chaque agent doit retourner un résultat standardisé.
        """
        raise NotImplementedError

    def build_result(
        self,
        prediction=None,
        probability=None,
        confidence=0.0,
        uncertainty=1.0,
        quality=0.0,
        details=None,
        error=None
    ):
        return {
            "agent_id": self.agent_id,
            "prediction": prediction,
            "probability": probability,
            "confidence": float(confidence),
            "uncertainty": float(uncertainty),
            "quality": float(quality),
            "details": details or {},
            "error": error
        }
