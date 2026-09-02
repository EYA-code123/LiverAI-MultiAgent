import numpy as np


class AdaptiveTrustManager:

    def __init__(
        self,
        alpha=0.30,
        beta=0.20,
        gamma=0.20,
        delta=0.15,
        epsilon=0.15
    ):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

        self.historical_performance = {}

        self.trust_history = {}

    def register_agent(
        self,
        agent_id,
        performance=0.5
    ):

        self.historical_performance[
            agent_id
        ] = self._clip(performance)

        self.trust_history.setdefault(
            agent_id,
            []
        )

    def get_historical_performance(
        self,
        agent_id
    ):

        return self.historical_performance.get(
            agent_id,
            0.5
        )

    def compute_trust(
        self,
        agent_id,
        confidence,
        uncertainty,
        quality,
        agreement=1.0,
        modality_availability=1.0
    ):

        historical = (
            self.get_historical_performance(
                agent_id
            )
        )

        confidence = self._clip(confidence)
        uncertainty = self._clip(uncertainty)
        quality = self._clip(quality)
        agreement = self._clip(agreement)
        modality_availability = self._clip(
            modality_availability
        )

        reliability = (
            confidence
            * (1.0 - uncertainty)
        )

        trust = (

            self.alpha * historical

            + self.beta * reliability

            + self.gamma * quality

            + self.delta * agreement

            + self.epsilon
            * modality_availability
        )

        trust = self._clip(trust)

        self.trust_history[
            agent_id
        ].append(trust)

        return trust

    def update_from_feedback(
        self,
        agent_id,
        correct,
        learning_rate=0.1
    ):

        previous = (
            self.get_historical_performance(
                agent_id
            )
        )

        target = 1.0 if correct else 0.0

        updated = (
            (1.0 - learning_rate) * previous
            + learning_rate * target
        )

        self.historical_performance[
            agent_id
        ] = self._clip(updated)

        return self.historical_performance[
            agent_id
        ]

    @staticmethod
    def _clip(value):

        return float(
            np.clip(
                value,
                0.0,
                1.0
            )
        )
