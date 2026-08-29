from abc import ABC, abstractmethod


class BaseAgent(ABC):

    def __init__(
        self,
        name,
        model_name,
        modality=None
    ):

        self.name = name
        self.model_name = model_name
        self.modality = modality

    @abstractmethod
    def predict(self, patient_data):

        pass

    def build_result(
        self,
        prediction,
        probability=None,
        probabilities=None,
        confidence=None,
        uncertainty=None,
        quality=None,
        details=None,
        recommended_action=None
    ):

        from orchestrator.schemas import AgentResult

        return AgentResult(

            agent=self.name,

            status="success",

            prediction=prediction,

            probability=probability,

            probabilities=probabilities,

            confidence=confidence,

            uncertainty=uncertainty,

            quality=quality,

            model=self.model_name,

            modality=self.modality,

            recommended_action=recommended_action,

            details=details or {}
        )
