from abc import ABC, abstractmethod


class BaseAgent(ABC):

    def __init__(self, agent_id, model=None):
        self.agent_id = agent_id
        self.model = model

    @abstractmethod
    def predict(self, X):
        """
        Returns prediction information for one or more samples.
        """
        pass

    def build_output(
        self,
        prediction,
        confidence,
        uncertainty=None,
        quality=None,
        explanation=None,
        recommendation=None
    ):
        return {
            "agent_id": self.agent_id,
            "prediction": prediction,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "quality": quality,
            "explanation": explanation,
            "recommendation": recommendation
        }
