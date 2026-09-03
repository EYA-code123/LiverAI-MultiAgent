from coordinator.coordinator import (
    LiverAICoordinator
)


class DummyAgent:

    def predict(
        self,
        data
    ):

        return {

            "prediction":
                "Positive",

            "probability":
                0.90,

            "class_probabilities": {

                "Negative":
                    0.10,

                "Positive":
                    0.90
            },

            "confidence":
                0.90,

            "uncertainty":
                0.10,

            "quality":
                0.95,

            "missing_data_ratio":
                0.05,

            "stability":
                0.90,

            "utility":
                0.90
        }


def test_full_pipeline():

    coordinator = (
        LiverAICoordinator()
    )

    coordinator.register_agent(

        agent_id=
            "AgentA",

        agent=
            DummyAgent(),

        task_type=
            "test_classification",

        modality=
            "clinical"
    )

    output = coordinator.run(

        patient_id=
            "P001",

        inputs={
            "AgentA":
                {"feature": 1}
        }
    )

    assert output[
        "status"
    ] == "completed"

    assert output[
        "decision"
    ] is not None

    assert output[
        "action"
    ] is not None

    assert output[
        "reasoning"
    ] is not None
