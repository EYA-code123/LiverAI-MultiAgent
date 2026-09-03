from coordinator.trust_manager import (
    TrustManager
)


def test_trust_range():

    manager = TrustManager()

    trust = manager.compute_trust(

        agent_id=
            "AgentA",

        confidence=
            0.90,

        uncertainty=
            0.10,

        quality=
            0.95,

        missing_data_ratio=
            0.05,

        agreement=
            0.90,

        stability=
            0.85,

        utility=
            0.90
    )

    assert 0.0 <= trust <= 1.0


def test_feedback_changes_performance():

    manager = TrustManager()

    manager.register_agent(
        "AgentA",
        0.50
    )

    before = manager.get_performance(
        "AgentA"
    )

    after = manager.update_from_outcome(
        "AgentA",
        True
    )

    assert after > before
