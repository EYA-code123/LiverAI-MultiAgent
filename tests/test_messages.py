from communication.message import AgentMessage


def test_agent_message():

    message = AgentMessage(

        patient_id="P001",

        agent_id="CirrhosisAgent",

        task_type=
            "cirrhosis_classification",

        prediction=
            "Decompensated",

        probability=
            0.85,

        confidence=
            0.85,

        uncertainty=
            0.15,

        quality=
            0.90,

        missing_data_ratio=
            0.10,

        trust=
            0.80
    )

    validation = (
        message.validate()
    )

    assert validation[
        "valid"
    ]

    data = (
        message.to_dict()
    )

    assert data[
        "agent_id"
    ] == "CirrhosisAgent"
