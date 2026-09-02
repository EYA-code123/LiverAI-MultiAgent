import numpy as np


class AdaptiveTrustManager:

    def __init__(self):

        self.historical_performance = {}
        self.trust_history = {}

        self.weights = {
            "historical": 0.25,
            "confidence": 0.20,
            "uncertainty": 0.15,
            "quality": 0.15,
            "agreement": 0.15,
            "availability": 0.10
        }

    def register_agent(
        self,
        agent_id,
        performance=0.5
    ):

        self.historical_performance[
            agent_id
        ] = float(
            np.clip(
                performance,
                0.0,
                1.0
            )
        )

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

        confidence = np.clip(
            confidence, 0, 1
        )

        uncertainty = np.clip(
            uncertainty, 0, 1
        )

        quality = np.clip(
            quality, 0, 1
        )

        agreement = np.clip(
            agreement, 0, 1
        )

        modality_availability = np.clip(
            modality_availability,
            0,
            1
        )

        trust = (

            self.weights["historical"]
            * historical

            +

            self.weights["confidence"]
            * confidence

            +

            self.weights["uncertainty"]
            * (1.0 - uncertainty)

            +

            self.weights["quality"]
            * quality

            +

            self.weights["agreement"]
            * agreement

            +

            self.weights["availability"]
            * modality_availability
        )

        trust = float(
            np.clip(
                trust,
                0,
                1
            )
        )

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

        target = (
            1.0
            if correct
            else 0.0
        )

        updated = (
            (1 - learning_rate)
            * previous
            +
            learning_rate
            * target
        )

        self.historical_performance[
            agent_id
        ] = float(
            np.clip(
                updated,
                0,
                1
            )
        )

        return self.historical_performance[
            agent_id
        ]
