# =============================================================================
# LiverAI-MultiAgent
# ADAPTIVE TRUST MANAGER
# =============================================================================

import numpy as np


class TrustManager:

    def __init__(self):

        self.historical_performance = {}

        self.default_performance = 0.5

    # -------------------------------------------------------------------------
    # REGISTER HISTORICAL PERFORMANCE
    # -------------------------------------------------------------------------

    def register_agent(
        self,
        agent_id,
        performance
    ):

        performance = self._clip(
            performance
        )

        self.historical_performance[
            agent_id
        ] = performance

    # -------------------------------------------------------------------------
    # GET HISTORICAL PERFORMANCE
    # -------------------------------------------------------------------------

    def get_historical_performance(
        self,
        agent_id
    ):

        return self.historical_performance.get(
            agent_id,
            self.default_performance
        )

    # -------------------------------------------------------------------------
    # COMPUTE PATIENT-SPECIFIC TRUST
    # -------------------------------------------------------------------------

    def compute_trust(
        self,
        agent_id,
        confidence=0.0,
        quality=0.0,
        uncertainty=1.0,
        missing_data_ratio=0.0,
        agreement=1.0
    ):

        historical = (
            self.get_historical_performance(
                agent_id
            )
        )

        confidence = self._clip(
            confidence
        )

        quality = self._clip(
            quality
        )

        uncertainty = self._clip(
            uncertainty
        )

        missing_data_ratio = self._clip(
            missing_data_ratio
        )

        agreement = self._clip(
            agreement
        )

        # Current reliability
        reliability = (
            0.40 * confidence
            + 0.25 * (1.0 - uncertainty)
            + 0.20 * quality
            + 0.15 * agreement
        )

        # Penalize missing data
        data_factor = (
            1.0 - missing_data_ratio
        )

        # Patient-specific trust
        trust = (
            0.50 * historical
            + 0.50 * reliability
        )

        trust *= data_factor

        return float(
            np.clip(
                trust,
                0.0,
                1.0
            )
        )

    # -------------------------------------------------------------------------
    # UPDATE FROM FEEDBACK
    # -------------------------------------------------------------------------

    def update_from_feedback(
        self,
        agent_id,
        correct,
        learning_rate=0.10
    ):

        old_value = (
            self.get_historical_performance(
                agent_id
            )
        )

        target = 1.0 if correct else 0.0

        new_value = (
            (1.0 - learning_rate) * old_value
            + learning_rate * target
        )

        self.historical_performance[
            agent_id
        ] = float(
            np.clip(
                new_value,
                0.0,
                1.0
            )
        )

        return self.historical_performance[
            agent_id
        ]

    # -------------------------------------------------------------------------
    # CLIP
    # -------------------------------------------------------------------------

    @staticmethod
    def _clip(value):

        try:
            value = float(value)
        except Exception:
            value = 0.0

        return float(
            np.clip(
                value,
                0.0,
                1.0
            )
        )
