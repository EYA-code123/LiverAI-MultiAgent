# =============================================================================
# LiverAI-MultiAgent
# FILE: coordinator/decision.py
# DECISION INTELLIGENCE ENGINE
# =============================================================================

from typing import Any


class DecisionEngine:
    """
    Decision Intelligence layer for LiverAI.

    This component does NOT replace the specialized models.

    It evaluates:

        - agent coverage
        - confidence
        - trust
        - uncertainty
        - conflicts
        - clinical reasoning
        - fused evidence

    The engine produces a structured system-level decision.
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        low_confidence_threshold: float = 0.55,
        high_confidence_threshold: float = 0.80,
        high_conflict_threshold: float = 0.50,
        minimum_agent_coverage: float = 0.50,
    ):

        self.low_confidence_threshold = (
            float(
                low_confidence_threshold
            )
        )

        self.high_confidence_threshold = (
            float(
                high_confidence_threshold
            )
        )

        self.high_conflict_threshold = (
            float(
                high_conflict_threshold
            )
        )

        self.minimum_agent_coverage = (
            float(
                minimum_agent_coverage
            )
        )

    # =========================================================================
    # MAIN DECISION
    # =========================================================================

    def decide(
        self,
        agent_results,
        conflicts=None,
        fused_results=None,
        clinical_reasoning=None,
    ):

        conflicts = (
            conflicts
            if conflicts is not None
            else []
        )

        fused_results = (
            fused_results
            if fused_results is not None
            else {}
        )

        clinical_reasoning = (
            clinical_reasoning
            if clinical_reasoning is not None
            else {}
        )

        # ---------------------------------------------------------------------
        # PROTECT AGAINST INVALID INPUT
        # ---------------------------------------------------------------------

        if agent_results is None:
            agent_results = []

        if isinstance(
            agent_results,
            dict,
        ):
            agent_results = list(
                agent_results.values()
            )

        # ---------------------------------------------------------------------
        # TOTAL AGENTS
        # ---------------------------------------------------------------------

        total_agents = len(
            agent_results
        )

        # ---------------------------------------------------------------------
        # SUCCESSFUL AGENTS
        # ---------------------------------------------------------------------

        successful_agents = []

        for result in agent_results:

            if result is None:
                continue

            status = self._get(
                result,
                "status",
                "success",
            )

            error = self._get(
                result,
                "error",
                None,
            )

            prediction = self._get(
                result,
                "prediction",
                None,
            )

            if (
                status in (
                    "success",
                    "completed",
                )
                and error is None
                and prediction is not None
            ):

                successful_agents.append(
                    result
                )

        failed_agents = (
            total_agents
            - len(successful_agents)
        )

        # ---------------------------------------------------------------------
        # COVERAGE
        # ---------------------------------------------------------------------

        if total_agents > 0:

            coverage = (
                len(successful_agents)
                /
                total_agents
            )

        else:

            coverage = 0.0

        # ---------------------------------------------------------------------
        # CONFIDENCE / TRUST / UNCERTAINTY
        # ---------------------------------------------------------------------

        trust_values = []
        confidence_values = []
        uncertainty_values = []
        quality_values = []

        for result in successful_agents:

            trust = self._safe_float(
                self._get(
                    result,
                    "trust",
                    0.0,
                )
            )

            confidence = self._safe_float(
                self._get(
                    result,
                    "confidence",
                    0.0,
                )
            )

            uncertainty = self._safe_float(
                self._get(
                    result,
                    "uncertainty",
                    1.0,
                )
            )

            quality = self._safe_float(
                self._get(
                    result,
                    "quality",
                    0.0,
                )
            )

            trust_values.append(
                self._clip(trust)
            )

            confidence_values.append(
                self._clip(confidence)
            )

            uncertainty_values.append(
                self._clip(uncertainty)
            )

            quality_values.append(
                self._clip(quality)
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

        average_quality = (
            sum(quality_values)
            /
            len(quality_values)
            if quality_values
            else 0.0
        )

        # ---------------------------------------------------------------------
        # CONFLICT SCORE
        # ---------------------------------------------------------------------

        conflict_strengths = []

        for conflict in conflicts:

            if not isinstance(
                conflict,
                dict,
            ):
                continue

            strength = self._safe_float(
                conflict.get(
                    "conflict_strength",
                    0.0,
                )
            )

            conflict_strengths.append(
                self._clip(strength)
            )

        if conflict_strengths:

            conflict_score = (
                sum(conflict_strengths)
                /
                len(conflict_strengths)
            )

        else:

            conflict_score = 0.0

        # ---------------------------------------------------------------------
        # CLINICAL REASONING
        # ---------------------------------------------------------------------

        clinical_prediction = None

        clinical_confidence = 0.0

        clinical_risk_score = None

        clinical_risk_level = None

        if isinstance(
            clinical_reasoning,
            dict,
        ):

            clinical_prediction = (
                clinical_reasoning.get(
                    "prediction"
                )
            )

            clinical_confidence = (
                self._safe_float(
                    clinical_reasoning.get(
                        "confidence",
                        clinical_reasoning.get(
                            "probability",
                            0.0,
                        ),
                    )
                )
            )

            clinical_confidence = (
                self._clip(
                    clinical_confidence
                )
            )

            clinical_risk_score = (
                clinical_reasoning.get(
                    "risk_score"
                )
            )

            clinical_risk_level = (
                clinical_reasoning.get(
                    "overall_risk"
                )
            )

            # ---------------------------------------------------------------
            # Some versions of the Clinical Reasoning Agent may store the
            # risk information inside unified_assessment.
            # ---------------------------------------------------------------

            unified = (
                clinical_reasoning.get(
                    "unified_assessment",
                    {}
                )
            )

            if isinstance(
                unified,
                dict,
            ):

                if clinical_risk_score is None:

                    clinical_risk_score = (
                        unified.get(
                            "risk_score"
                        )
                    )

                if clinical_risk_level is None:

                    clinical_risk_level = (
                        unified.get(
                            "overall_risk"
                        )
                    )

        # ---------------------------------------------------------------------
        # ADDITIONAL TEST / REVIEW LOGIC
        # ---------------------------------------------------------------------

        request_additional_tests = False

        reasons = []

        if coverage < (
            self.minimum_agent_coverage
        ):

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

        # ---------------------------------------------------------------------
        # DECISION CONFIDENCE
        # ---------------------------------------------------------------------

        decision_confidence = (

            0.30
            * average_trust

            +

            0.30
            * average_confidence

            +

            0.20
            * coverage

            +

            0.20
            * (
                1.0
                - conflict_score
            )
        )

        decision_confidence = (
            self._clip(
                decision_confidence
            )
        )

        # ---------------------------------------------------------------------
        # CLINICAL REASONING CONTRIBUTION
        # ---------------------------------------------------------------------

        if clinical_prediction is not None:

            # Clinical reasoning is additional evidence.
            #
            # We do not allow it to completely override the other agents.

            decision_confidence = (
                0.75
                * decision_confidence
                +
                0.25
                * clinical_confidence
            )

            decision_confidence = (
                self._clip(
                    decision_confidence
                )
            )

        # ---------------------------------------------------------------------
        # FINAL DECISION
        # ---------------------------------------------------------------------

        if not successful_agents:

            decision = (
                "insufficient_evidence"
            )

        elif coverage < (
            self.minimum_agent_coverage
        ):

            decision = (
                "insufficient_coverage"
            )

        elif conflict_score >= (
            self.high_conflict_threshold
        ):

            decision = (
                "conflict_review_required"
            )

        elif clinical_prediction is not None:

            if clinical_confidence >= (
                self.high_confidence_threshold
            ):

                decision = (
                    "clinical_reasoning_supported"
                )

            elif request_additional_tests:

                decision = (
                    "clinical_reasoning_with_review"
                )

            else:

                decision = (
                    "clinical_reasoning_supported"
                )

        elif decision_confidence >= (
            self.high_confidence_threshold
        ):

            decision = (
                "evidence_supported"
            )

        elif request_additional_tests:

            decision = (
                "review_required"
            )

        else:

            decision = (
                "evidence_supported"
            )

        # ---------------------------------------------------------------------
        # RISK SCORE
        # ---------------------------------------------------------------------

        if clinical_risk_score is not None:

            risk_score = self._clip(
                clinical_risk_score
            )

        else:

            risk_score = self._clip(
                1.0
                - decision_confidence
            )

        # ---------------------------------------------------------------------
        # FINAL STATUS
        # ---------------------------------------------------------------------

        if not successful_agents:

            status = (
                "insufficient_evidence"
            )

        elif request_additional_tests:

            status = (
                "review_required"
            )

        else:

            status = (
                "decision_ready"
            )

        # ---------------------------------------------------------------------
        # WARNING
        # ---------------------------------------------------------------------

        warning = None

        if conflicts:

            warning = (
                "Prediction conflicts were detected "
                "between compatible agents."
            )

        # ---------------------------------------------------------------------
        # EXPLANATION
        # ---------------------------------------------------------------------

        explanation = (
            f"{len(successful_agents)}/"
            f"{total_agents} agents completed successfully. "
            f"Coverage={coverage:.3f}. "
            f"Mean confidence={average_confidence:.3f}. "
            f"Mean trust={average_trust:.3f}. "
            f"Mean uncertainty={average_uncertainty:.3f}. "
            f"Conflict score={conflict_score:.3f}. "
            f"Clinical reasoning confidence="
            f"{clinical_confidence:.3f}. "
            f"System decision: {decision}."
        )

        # ---------------------------------------------------------------------
        # FINAL OBJECT
        # ---------------------------------------------------------------------

        return {
            "status": status,

            # IMPORTANT:
            # This is the key expected by the orchestrator.

            "decision": decision,

            "confidence": decision_confidence,

            "risk_score": risk_score,

            "coverage": coverage,

            "successful_agents":
                len(successful_agents),

            "total_agents":
                total_agents,

            "failed_agents":
                failed_agents,

            "mean_confidence":
                average_confidence,

            "mean_trust":
                average_trust,

            "mean_uncertainty":
                average_uncertainty,

            "mean_quality":
                average_quality,

            "conflict_score":
                conflict_score,

            "num_conflicts":
                len(conflicts),

            "clinical_prediction":
                clinical_prediction,

            "clinical_confidence":
                clinical_confidence,

            "clinical_risk_score":
                clinical_risk_score,

            "clinical_risk_level":
                clinical_risk_level,

            "request_additional_tests":
                request_additional_tests,

            "additional_test_reasons":
                reasons,

            "conflicts":
                conflicts,

            "fused_evidence":
                fused_results,

            "clinical_reasoning":
                clinical_reasoning,

            "warning":
                warning,

            "explanation":
                explanation,
        }

    # =========================================================================
    # SAFE ACCESS
    # =========================================================================

    @staticmethod
    def _get(
        obj,
        key,
        default=None,
    ):

        if isinstance(
            obj,
            dict,
        ):

            return obj.get(
                key,
                default,
            )

        return getattr(
            obj,
            key,
            default,
        )

    # =========================================================================
    # SAFE FLOAT
    # =========================================================================

    @staticmethod
    def _safe_float(
        value,
        default=0.0,
    ):

        try:

            return float(value)

        except (
            TypeError,
            ValueError,
        ):

            return default

    # =========================================================================
    # CLIP
    # =========================================================================

    @staticmethod
    def _clip(
        value,
    ):

        try:

            value = float(value)

        except (
            TypeError,
            ValueError,
        ):

            value = 0.0

        return max(
            0.0,
            min(
                1.0,
                value,
            )
        )
