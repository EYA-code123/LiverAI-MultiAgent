import numpy as np


class TrustManager:

    def __init__(self):

        # Performance historique des agents.
        # Ces valeurs seront remplacées par les métriques
        # réellement obtenues pendant l'évaluation.
        self.historical_performance = {}

    def register_agent(self, agent_id, performance):

        self.historical_performance[agent_id] = float(
            np.clip(performance, 0.0, 1.0)
        )

    def get_historical_performance(self, agent_id):

        return self.historical_performance.get(
            agent_id,
            0.5
        )

    def compute_trust(
        self,
        agent_id,
        confidence,
        quality,
        uncertainty
    ):

        historical = self.get_historical_performance(agent_id)

        confidence = float(
            np.clip(confidence, 0.0, 1.0)
        )

        quality = float(
            np.clip(quality, 0.0, 1.0)
        )

        uncertainty = float(
            np.clip(uncertainty, 0.0, 1.0)
        )

        current_reliability = (
            confidence *
            (1.0 - uncertainty)
        )

        trust = (
            historical *
            current_reliability *
            quality
        )

        return float(
            np.clip(trust, 0.0, 1.0)
        ) 
