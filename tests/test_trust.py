# tests/test_trust.py

from coordinator.trust_manager import TrustManager


def test_trust_between_zero_and_one():

    manager = TrustManager()

    trust = manager.compute_trust(

        agent_id="CirrhosisAgent",

        confidence=0.9,

        uncertainty=0.1,

        quality=0.95,

        missing_data_ratio=0.05,

        agreement=0.9,

        stability=0.8,

        utility=0.9,

        modality_available=True
    )

    assert 0.0 <= trust <= 1.0
