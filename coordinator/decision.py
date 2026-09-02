# =============================================================================
# LiverAI-MultiAgent
# DECISION INTELLIGENCE ENGINE
# =============================================================================


class DecisionEngine:
    """
    Converts coordinated agent evidence into a final decision.

    This engine does NOT replace the specialized models.

    It evaluates:

        - agent coverage
        - trust
        - confidence
        - uncertainty
        - conflicts
        - clinical reasoning
        - evidence quality

    It can also determine whether additional information would be useful.
    """

    def __init__(
        self,
        low_confidence_threshold: float = 0.55,
        high_confidence_threshold: float = 0.80,
        high_conflict_threshold: float = 0.50,
        minimum_agent_coverage: float = 0.50,
    ):

        self.low_confidence_threshold = (
            low_confidence_threshold
        )

        self.high_confidence_threshold = (
            high_confidence_threshold
        )

        self.high_conflict_threshold = (
            high_conflict_threshold
        )

        self.minimum_agent_coverage = (
            minimum_agent_coverage
        )

    # -------------------------------------------------------------------------
    # MAIN DECISION
    # -------------------------------------------------------------------------

    def decide(
        self,
        agent_results,
        conflicts=None,
        fused_results=None,
        clinical_reasoning=None,
    ):

        conflicts = conflicts or []
        fused_results = fused_results or {}
        clinical_reasoning = (
            clinical_reasoning or {}
        )

        total_agents = len(
            agent_results
        )

        successful_agents = [
            result
            for result in agent_results
            if self._get(
                result,
                "error",
                None
            ) is None
            and
            self._get(
                result,
                "prediction",
                None
            ) is not None
        ]

        failed_agents = (
            total_agents
            -
            len(successful_agents)
        )

        # -------------------------------------------------------------
        # AGENT COVERAGE
        # -------------------------------------------------------------

        if total_agents > 0:

            coverage = (
                len(successful_agents)
                /
                total_agents
            )

        else:

            coverage = 0.0

        # -------------------------------------------------------------
        # TRUST / CONFIDENCE
        # -------------------------------------------------------------

        trust_values = []

        confidence_values = []

        uncertainty_values = []

        for result in successful_agents:

            trust = self._get(
                result,
                "trust",
                0.0
            )

            confidence = self._get(
                result,
                "confidence",
                0.0
            )

            uncertainty = self._get(
                result,
                "uncertainty",
                1.0
            )

            trust_values.append(
                float(trust)
            )

            confidence_values.append(
                float(confidence)
            )

            uncertainty_values.append(
                float(uncertainty)
            )

        average_trust = (
            sum(trust_values)
            /
            len(trust_values)
            if trust_values
            else 0.0
        )

        average_confidence = (
            sum(confidence_values)
            /
            len(confidence_values)
            if confidence_values
            else 0.0
        )

        average_uncertainty = (
            sum(uncertainty_values)
            /
            len(uncertainty_values)
            if uncertainty_values
            else 1.0
        )

        # -------------------------------------------------------------
        # CONFLICT SCORE
        # -------------------------------------------------------------

        if conflicts:

            conflict_strengths = [
                float(
                    conflict.get(
                        "conflict_strength",
                        0.0
                    )
                )
                for conflict in conflicts
            ]

            conflict_score = (
                sum(conflict_strengths)
                /
                len(conflict_strengths)
            )

        else:

            conflict_score = 0.0

        # -------------------------------------------------------------
        # CLINICAL REASONING
        # -------------------------------------------------------------

        clinical_prediction = None
        clinical_confidence = 0.0
        risk_score = None
        risk_level = None

        if isinstance(
            clinical_reasoning,
            dict
        ):

            clinical_prediction = (
                clinical_reasoning.get(
                    "prediction"
                )
            )

            clinical_confidence = float(
                clinical_reasoning.get(
                    "confidence",
                    clinical_reasoning.get(
                        "probability",
                        0.0
                    )
                ) or 0.0
            )

            risk_score = (
                clinical_reasoning.get(
                    "risk_score"
                )
            )

            risk_level = (
                clinical_reasoning.get(
                    "overall_risk"
                )
            )

            # Current ClinicalReasoningAgent
            # may store these inside unified_assessment.
            unified = clinical_reasoning.get(
                "unified_assessment",
                {}
            )

            if isinstance(
                unified,
                dict
            ):

                if risk_score is None:

                    risk_score = unified.get(
                        "risk_score"
                    )

                if risk_level is None:

                    risk_level = unified.get(
                        "overall_risk"
                    )

        # -------------------------------------------------------------
        # REQUEST ADDITIONAL TESTS
        # -------------------------------------------------------------

        reasons = []

        request_additional_tests = False

        if coverage < self.minimum_agent_coverage:

            request_additional_tests = True

            reasons.append(
                "Insufficient agent coverage."
            )

        if average_confidence < (
            self.low_confidence_threshold
        ):

            request_additional_tests = True

            reasons.append(
                "Average prediction confidence is low."
            )

        if average_uncertainty > 0.50:

            request_additional_tests = True

            reasons.append(
                "Prediction uncertainty is high."
            )

        if conflict_score >= (
            self.high_conflict_threshold
        ):

            request_additional_tests = True

            reasons.append(
                "Strong disagreement detected between compatible agents."
            )

        # -------------------------------------------------------------
        # DECISION CONFIDENCE
        # -------------------------------------------------------------

        decision_confidence = (
            0.30 * average_trust
            +
            0.30 * average_confidence
            +
            0.20 * coverage
            +
            0.20 * (
                1.0 - conflict_score
            )
        )

        decision_confidence = max(
            0.0,
            min(
                1.0,
                decision_confidence
            )
        )

        # -------------------------------------------------------------
        # FINAL STATUS
        # -------------------------------------------------------------

        if not successful_agents:

            status = "insufficient_evidence"

        elif request_additional_tests:

            status = "review_required"

        else:

            status = "decision_ready"

        # -------------------------------------------------------------
        # WARNING
        # -------------------------------------------------------------

        warning = None

        if conflicts:

            warning = (
                "Prediction conflicts were detected "
                "between compatible agents."
            )

        # -------------------------------------------------------------
        # FINAL OBJECT
        # -------------------------------------------------------------

        return {

            "status":
                status,

            "num_agents":
                total_agents,

            "successful_agents":
                len(successful_agents),

            "failed_agents":
                failed_agents,

            "agent_coverage":
                coverage,

            "average_trust":
                average_trust,

            "average_confidence":
                average_confidence,

            "average_uncertainty":
                average_uncertainty,

            "conflict_score":
                conflict_score,

            "num_conflicts":
                len(conflicts),

            "decision_confidence":
                decision_confidence,

            "clinical_prediction":
                clinical_prediction,

            "clinical_confidence":
                clinical_confidence,

            "risk_score":
                risk_score,

            "risk_level":
                risk_level,

            "request_additional_tests":
                request_additional_tests,

            "additional_test_reasons":
                reasons,

            "conflicts":
                conflicts,

            "fused_evidence":
                fused_results,

            "warning":
                warning,
        }

    # -------------------------------------------------------------------------
    # UTILITY
    # -------------------------------------------------------------------------

    @staticmethod
    def _get(
        obj,
        key,
        default=None
    ):

        if isinstance(obj, dict):

            return obj.get(
                key,
                default
            )

        return getattr(
            obj,
            key,
            default
        )
