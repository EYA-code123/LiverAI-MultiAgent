from collections import defaultdict
import numpy as np


class TrustManager:

    def __init__(
        self,
        historical_weight=0.30,
        current_weight=0.70
    ):

        self.historical_weight = float(
            historical_weight
        )

        self.current_weight = float(
            current_weight
        )

        if abs(
            self.historical_weight
            + self.current_weight
            - 1.0
        ) > 1e-6:

            raise ValueError(
                "Trust weights must sum to 1."
            )

        self.historical_performance = {}

        self.trust_history = (
            defaultdict(list)
        )

        self.feedback_history = (
            defaultdict(list)
        )

    # =========================================================
    # UTILITY
    # =========================================================

    @staticmethod
    def clip(
        value
    ):

        return float(
            np.clip(
                float(value),
                0.0,
                1.0
            )
        )

    # =========================================================
    # REGISTER AGENT
    # =========================================================

    def register_agent(
        self,
        agent_id,
        performance=0.5
    ):

        self.historical_performance[
            agent_id
        ] = self.clip(
            performance
        )

    # =========================================================
    # CURRENT RELIABILITY
    # =========================================================

    def compute_reliability(
        self,
        confidence,
        uncertainty,
        quality,
        missing_data_ratio,
        agreement,
        stability,
        utility,
        modality_available=True
    ):

        confidence = self.clip(
            confidence
        )

        uncertainty = self.clip(
            uncertainty
        )

        quality = self.clip(
            quality
        )

        missing_data_ratio = self.clip(
            missing_data_ratio
        )

        agreement = self.clip(
            agreement
        )

        stability = self.clip(
            stability
        )

        utility = self.clip(
            utility
        )

        modality = (
            1.0
            if modality_available
            else 0.0
        )

        reliability = (

            0.25 * confidence

            + 0.20 * (
                1.0 - uncertainty
            )

            + 0.20 * quality

            + 0.10 * (
                1.0 - missing_data_ratio
            )

            + 0.10 * agreement

            + 0.10 * stability

            + 0.05 * utility
        )

        reliability *= (
            0.80
            + 0.20 * modality
        )

        return self.clip(
            reliability
        )

    # =========================================================
    # TRUST
    # =========================================================

    def compute_trust(
        self,
        agent_id,
        confidence,
        uncertainty,
        quality,
        missing_data_ratio,
        agreement=0.5,
        stability=0.5,
        utility=0.5,
        modality_available=True
    ):

        if agent_id not in (
            self.historical_performance
        ):

            self.register_agent(
                agent_id
            )

        historical = (
            self.historical_performance[
                agent_id
            ]
        )

        reliability = (
            self.compute_reliability(

                confidence=confidence,

                uncertainty=uncertainty,

                quality=quality,

                missing_data_ratio=
                    missing_data_ratio,

                agreement=agreement,

                stability=stability,

                utility=utility,

                modality_available=
                    modality_available
            )
        )

        trust = (

            self.historical_weight
            * historical

            +

            self.current_weight
            * reliability
        )

        trust = self.clip(
            trust
        )

        self.trust_history[
            agent_id
        ].append(
            trust
        )

        return trust

    # =========================================================
    # FEEDBACK
    # =========================================================

    def update_from_outcome(
        self,
        agent_id,
        correct,
        learning_rate=0.10
    ):

        if agent_id not in (
            self.historical_performance
        ):

            self.register_agent(
                agent_id
            )

        old = (
            self.historical_performance[
                agent_id
            ]
        )

        target = (
            1.0
            if bool(correct)
            else 0.0
        )

        new = (

            (1.0 - learning_rate)
            * old

            +

            learning_rate
            * target
        )

        new = self.clip(
            new
        )

        self.historical_performance[
            agent_id
        ] = new

        self.feedback_history[
            agent_id
        ].append({

            "correct":
                bool(correct),

            "new_performance":
                new
        })

        return new

    # =========================================================
    # HISTORY
    # =========================================================

    def get_trust_history(
        self,
        agent_id
    ):

        return list(
            self.trust_history.get(
                agent_id,
                []
            )
        )

    def get_performance(
        self,
        agent_id
    ):

        return float(
            self.historical_performance.get(
                agent_id,
                0.5
            )
        )
