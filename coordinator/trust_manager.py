class TrustManager:

    def __init__(self):

        self.historical_performance = {}

    def register_agent(
        self,
        agent_name,
        performance
    ):

        self.historical_performance[
            agent_name
        ] = performance

    def compute_trust(
        self,
        agent_name,
        confidence=0.0,
        quality=1.0
    ):

        historical = self.historical_performance.get(
            agent_name,
            0.5
        )

        trust = (
            0.5 * historical
            +
            0.3 * confidence
            +
            0.2 * quality
        )

        return max(
            0.0,
            min(1.0, trust)
        )
