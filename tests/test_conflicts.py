# tests/test_conflicts.py

from coordinator.conflict_detector import (
    ConflictDetector
)


def test_conflict_detected():

    detector = ConflictDetector()

    results = [

        {
            "agent":
                "AgentA",

            "task_type":
                "cirrhosis_classification",

            "prediction":
                "Compensated",

            "confidence":
                0.80,

            "status":
                "success"
        },

        {
            "agent":
                "AgentB",

            "task_type":
                "cirrhosis_classification",

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
