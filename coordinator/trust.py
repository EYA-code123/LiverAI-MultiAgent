"""
Adaptive Trust Intelligence
============================

Patient-specific trust computation.

Trust is not a fixed weight.
It changes according to:

- historical performance
- confidence
- uncertainty
- data quality
- missing data
- modality availability
- agreement
- stability
- utility
"""

import numpy as np


class TrustManager:

    def __init__(
        self,
        historical_weight=0.30,
        current_weight=0.70
    ):

        self.historical_weight = historical_weight
        self.current_weight = current_weight

        self.historical_performance = {}
        self.trust_history = {}

    @staticmethod
    def clip(value):
        return float(
            np.clip(
                float(value),
                0.0,
                1.0
            )
        )

    def register_agent(
        self,
        agent_id,
        performance
    ):

        self.historical_performance[
            agent_id
        ] = self.clip(performance)

    def update_historical_performance(
        self,
        agent_id,
        outcome_correct,
        learning_rate=0.10
    ):

        old = self.historical_performance.get(
            agent_id,
            0.5
        )

        new_value = (
            (1.0 - learning_rate) * old
            + learning_rate * float(outcome_correct)
        )

        self.historical_performance[
            agent_id
        ] = self.clip(new_value)

        return self.historical_performance[
            agent_id
        ]

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
        quality,
        uncertainty,
        missing_data_ratio=0.0,
        agreement=0.5,
        stability=0.5,
        utility=0.5,
        modality_available=True
    ):

        historical = self.get_historical_performance(
            agent_id
        )

        confidence = self.clip(confidence)
        quality = self.clip(quality)
        uncertainty = self.clip(uncertainty)
        missing_data_ratio = self.clip(
            missing_data_ratio
        )
        agreement = self.clip(agreement)
        stability = self.clip(stability)
        utility = self.clip(utility)

        modality_score = (
            1.0
            if modality_available
            else 0.0
        )

        # Current reliability
        reliability = (
            0.25 * confidence
            + 0.20 * (1.0 - uncertainty)
            + 0.20 * quality
            + 0.10 * (1.0 - missing_data_ratio)
            + 0.10 * agreement
            + 0.10 * stability
            + 0.05 * utility
        )

        reliability *= (
            0.80 + 0.20 * modality_score
        )

        trust = (
            self.historical_weight
            * historical
            +
            self.current_weight
            * reliability
        )

        trust = self.clip(trust)

        self.trust_history.setdefault(
            agent_id,
            []
        ).append(trust)

        return trust

    def compute_message_trust(
        self,
        message,
        agreement=0.5,
        stability=0.5,
        utility=0.5,
        modality_available=True
    ):

        trust = self.compute_trust(
            agent_id=message.agent_id,
            confidence=message.confidence,
            quality=message.quality,
            uncertainty=message.uncertainty,
            missing_data_ratio=message.missing_data_ratio,
            agreement=agreement,
            stability=stability,
            utility=utility,
            modality_available=modality_available
        )

        message.trust = trust

        message.agreement = agreement
        message.stability = stability
        message.utility = utility

        return message

    def get_trust_history(
        self,
        agent_id
    ):

        return self.trust_history.get(
            agent_id,
            []
        )
