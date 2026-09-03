from coordinator.decision import (
    DecisionEngine
)


def test_high_confidence_decision():

    engine = DecisionEngine()

    results = [

        {

            "agent_id":
                "AgentA",

            "task_type":
                "cirrhosis",

            "prediction":
                "Decompensated",

            "confidence":
                0.90,

            "trust":
                0.90,

            "quality":
                0.95,

            "status":
                "success"
        },

        {

            "agent_id":
                "AgentB",

            "task_type":
                "cirrhosis",

            "prediction":
                "Decompensated",

            "confidence":
                0.85,

            "trust":
                0.85,

            "quality":
                0.90,

            "status":
                "success"
        }
    ]

    decision = engine.decide(

        results=
            results,

        conflicts=[],

        reasoning={
            "prediction":
                "Decompensated"
        }
    )

    assert decision[
        "decision_level"
    ] == "HIGH"
