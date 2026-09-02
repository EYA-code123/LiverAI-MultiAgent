# =============================================================================
# LiverAI-MultiAgent
# ADAPTIVE TRUST MANAGER
# =============================================================================

import numpy as np


class TrustManager:
    """
    Computes dynamic trust for each agent.

    Trust combines:

        historical performance
        current confidence
        uncertainty
        data quality
        missing-data ratio

    Formula:

        reliability =
            confidence
            * (1 - uncertainty)
            * quality
            * (1 - missing_ratio)

        trust =
            historical_weight * historical
            +
            current_weight * reliability
    """

    def __init__(
        self,
        default_trust: float = 0.5,
        historical_weight: float = 0.40,
        current_weight: float = 0.60,
    ):

        self.default_trust = float(
            np.clip(default_trust, 0.0, 1.0)
        )

        self.historical_weight = float(
            np.clip(historical_weight, 0.0, 1.0)
        )

        self.current_weight = float(
            np.clip(current_weight, 0.0, 1.0)
        )

        total = (
            self.historical_weight
            + self.current_weight
        )

        if total <= 0:
            self.historical_weight = 0.4
            self.current_weight = 0.6

        else:
            self.historical_weight /= total
            self.current_weight /= total

        self.historical_performance = {}

    # -------------------------------------------------------------------------
    # REGISTER
    # -------------------------------------------------------------------------

    def register_agent(
        self,
        agent_id,
        performance
    ):

        self.historical_performance[
            str(agent_id)
        ] = float(
            np.clip(
                performance,
                0.0,
                1.0
            )
        )

    # -------------------------------------------------------------------------
    # GET HISTORICAL
    # -------------------------------------------------------------------------

    def get_historical_performance(
        self,
        agent_id
    ):

        return self.historical_performance.get(
            str(agent_id),
            self.default_trust
        )

    # -------------------------------------------------------------------------
    # COMPUTE TRUST
    # -------------------------------------------------------------------------

    def compute_trust(
        self,
        agent_id,
        confidence,
        quality,
        uncertainty,
        missing_data_ratio=0.0,
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

        # -------------------------------------------------------------
        # CURRENT RELIABILITY
        # -------------------------------------------------------------

        current_reliability = (
            confidence
            * (1.0 - uncertainty)
            * quality
            * (1.0 - missing_data_ratio)
        )

        # -------------------------------------------------------------
        # ADAPTIVE TRUST
        # -------------------------------------------------------------

        trust = (
            self.historical_weight
            * historical
            +
            self.current_weight
            * current_reliability
        )

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
        correct: bool,
        learning_rate: float = 0.10
    ):

        agent_id = str(agent_id)

        old_value = (
            self.get_historical_performance(
                agent_id
            )
        )

        target = 1.0 if correct else 0.0

        learning_rate = self._clip(
            learning_rate
        )

        new_value = (
            (1.0 - learning_rate)
            * old_value
            +
            learning_rate
            * target
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
    # BATCH TRUST
    # -------------------------------------------------------------------------

    def evaluate_result(self, result):
        """
        Compute trust directly from an AgentResult-like object
        or dictionary.
        """

        if isinstance(result, dict):

            agent_id = result.get(
                "agent_id",
                result.get("agent", "unknown")
            )

            confidence = result.get(
                "confidence",
                0.0
            )

            quality = result.get(
                "quality",
                0.0
            )

            uncertainty = result.get(
                "uncertainty",
                1.0
            )

            missing_ratio = result.get(
                "missing_data_ratio",
                0.0
            )

        else:

            agent_id = result.agent_id
            confidence = result.confidence
            quality = result.quality
            uncertainty = result.uncertainty
            missing_ratio = getattr(
                result,
                "missing_data_ratio",
                0.0
            )

        return self.compute_trust(
            agent_id=agent_id,
            confidence=confidence,
            quality=quality,
            uncertainty=uncertainty,
            missing_data_ratio=missing_ratio,
        )

    # -------------------------------------------------------------------------
    # UTILITY
    # -------------------------------------------------------------------------

    @staticmethod
    def _clip(value):

        try:
            value = float(value)

        except (
            TypeError,
            ValueError
        ):
            value = 0.0

        return float(
            np.clip(
                value,
                0.0,
                1.0
            )
        )
