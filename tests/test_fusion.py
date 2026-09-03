# tests/test_fusion.py

from coordinator.adaptive_fusion import AdaptiveFusion


def test_same_task_fusion():

    fusion = AdaptiveFusion()

    results = [

        {
            "agent": "AgentA",
            "task_type":
                "cirrhosis_classification",

            "prediction":
                "Decompensated",

            "class_probabilities": {
                "Compensated": 0.10,
                "Decompensated": 0.80,
                "No_Cirrhosis": 0.10
            },

            "confidence": 0.80,
            "uncertainty": 0.20,
            "quality": 0.90,
            "trust": 0.90,
            "agreement": 0.80,
            "stability": 0.80,
            "utility": 0.90
        },

        {
            "agent": "AgentB",
            "task_type":
                "cirrhosis_classification",

            "prediction":
                "Decompensated",

            "class_probabilities": {
                "Compensated": 0.15,
                "Decompensated": 0.70,
                "No_Cirrhosis": 0.15
            },

            "confidence": 0.70,
            "uncertainty": 0.30,
            "quality": 0.85,
            "trust": 0.80,
            "agreement": 0.80,
            "stability": 0.75,
            "utility": 0.80
        }
    ]

    output = fusion.fuse(
        results
    )

    assert output[
        "status"
    ] == "success"

    assert (
        "cirrhosis_classification"
        in output[
            "same_task_fusion"
        ]
    )
