from coordinator.adaptive_fusion import (
    AdaptiveFusion
)


def test_adaptive_fusion():

    fusion = AdaptiveFusion()

    results = [

        {

            "agent_id":
                "AgentA",

            "task_type":
                "cirrhosis",

            "prediction":
                "Decompensated",

            "class_probabilities": {

                "Compensated":
                    0.10,

                "Decompensated":
                    0.80,

                "No_Cirrhosis":
                    0.10
            },

            "confidence":
                0.80,

            "uncertainty":
                0.20,

            "quality":
                0.90,

            "trust":
                0.90,

            "agreement":
                0.80,

            "stability":
                0.80,

            "utility":
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
                "Decompensated",

            "class_probabilities": {

                "Compensated":
                    0.15,

                "Decompensated":
                    0.70,

                "No_Cirrhosis":
                    0.15
            },

            "confidence":
                0.70,

            "uncertainty":
                0.30,

            "quality":
                0.85,

            "trust":
                0.80,

            "agreement":
                0.80,

            "stability":
                0.75,

            "utility":
                0.80,

            "status":
                "success"
        }
    ]

    output = fusion.fuse(
        results
    )

    assert output[
        "status"
    ] == "success"

    assert (
        "cirrhosis"
        in output[
            "same_task_fusion"
        ]
    )
