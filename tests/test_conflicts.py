from coordinator.conflict_detector import (
    ConflictDetector
)

from coordinator.conflict_resolver import (
    ConflictResolver
)


def test_conflict_detection():

    detector = ConflictDetector()

    results = [

        {

            "agent_id":
                "AgentA",

            "task_type":
                "cirrhosis",

            "prediction":
                "Compensated",

            "confidence":
                0.80,

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
                0.75,

            "status":
                "success"
        }
    ]

    conflicts = detector.detect(
        results
    )

    assert len(
        conflicts
    ) == 1


def test_conflict_resolution():

    resolver = ConflictResolver()

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

            "stability":
                0.90,

            "status":
                "success"
        },

        {

            "agent_id":
                "AgentB",

            "task_type":
                "cirrhosis",

            "prediction":
                "Compensated",

            "confidence":
                0.50,

            "trust":
                0.40,

            "quality":
                0.70,

            "stability":
                0.50,

            "status":
                "success"
        }
    ]

    output = resolver.resolve(

        task_type=
            "cirrhosis",

        results=
            results,

        conflicts=[
            {
                "task_type":
                    "cirrhosis",
                "conflict_strength":
                    0.70
            }
        ]
    )

    assert output[
        "prediction"
    ] == "Decompensated"
