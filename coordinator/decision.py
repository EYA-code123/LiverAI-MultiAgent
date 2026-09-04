# ================================================================
# DECISION ENGINE
# ================================================================
# Produces a final system-level decision from:
#
#   1. Specialized agent outputs
#   2. Conflict detection
#   3. Clinical reasoning
#
# IMPORTANT:
# The final "risk_score" generated here is a SYSTEM confidence /
# reliability score. It is NOT a medical probability.
# ================================================================

from typing import Any, Dict, List
import math


class DecisionEngine:
    """
    Final decision layer for the LiverAI multi-agent architecture.
    """

    def __init__(
        self,
        minimum_coverage: float = 0.50,
        minimum_confidence: float = 0.50,
        minimum_trust: float = 0.50,
    ):

        self.minimum_coverage = float(
            minimum_coverage
        )

        self.minimum_confidence = float(
            minimum_confidence
        )

        self.minimum_trust = float(
            minimum_trust
        )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def decide(
        self,
        results: List[Dict[str, Any]],
        conflicts=None,
        reasoning=None,
    ) -> Dict[str, Any]:
        """
        Produce the final system decision.

        Parameters
        ----------
        results : list
            Normalized agent result dictionaries.

        conflicts : list
            Conflicts detected by ConflictDetector.

        reasoning : dict
            Clinical reasoning output.

        Returns
        -------
        dict
            Final decision.
        """

        # --------------------------------------------------------
        # Normalize inputs
        # --------------------------------------------------------

        results = self._normalize_results(
            results
        )

        conflicts = self._normalize_conflicts(
            conflicts
        )

        reasoning = self._normalize_reasoning(
            reasoning
        )

        # --------------------------------------------------------
        # No results
        # --------------------------------------------------------

        if not results:

            return {
                "status": "unavailable",
                "decision": "insufficient_data",
                "message": (
                    "No agent results are available."
                ),
                "coverage": 0.0,
                "mean_confidence": 0.0,
                "mean_trust": 0.0,
                "risk_score": 1.0,
                "confidence": 0.0,
                "conflicts": [],
                "clinical_reasoning": reasoning,
            }

        # --------------------------------------------------------
        # Analyze successful agents
        # --------------------------------------------------------

        successful = []

        for result in results:

            status = str(
                result.get(
                    "status",
                    ""
                )
            ).lower()

            if status == "success":
                successful.append(
                    result
                )

        total_agents = len(
            results
        )

        successful_agents = len(
            successful
        )

        coverage = (
            successful_agents / total_agents
            if total_agents > 0
            else 0.0
        )

        # --------------------------------------------------------
        # Aggregate confidence / trust / quality
        # --------------------------------------------------------

        mean_confidence = self._mean(
            [
                self._clip01(
                    self._safe_float(
                        r.get(
                            "confidence"
                        ),
                        0.0
                    )
                )
                for r in successful
            ]
        )

        mean_trust = self._mean(
            [
                self._clip01(
                    self._safe_float(
                        r.get(
                            "trust"
                        ),
                        0.0
                    )
                )
                for r in successful
            ]
        )

        mean_quality = self._mean(
            [
                self._clip01(
                    self._safe_float(
                        r.get(
                            "quality"
                        ),
                        0.0
                    )
                )
                for r in successful
            ]
        )

        # --------------------------------------------------------
        # Conflict score
        # --------------------------------------------------------

        conflict_score = self._calculate_conflict_score(
            conflicts,
            successful_agents
        )

        # --------------------------------------------------------
        # Clinical reasoning confidence
        # --------------------------------------------------------

        clinical_confidence = self._clinical_confidence(
            reasoning
        )

        # --------------------------------------------------------
        # System confidence
        # --------------------------------------------------------

        system_confidence = self._calculate_system_confidence(
            coverage=coverage,
            mean_confidence=mean_confidence,
            mean_trust=mean_trust,
            mean_quality=mean_quality,
            conflict_score=conflict_score,
            clinical_confidence=clinical_confidence,
        )

        # --------------------------------------------------------
        # System risk
        # --------------------------------------------------------

        risk_score = self._clip01(
            1.0 - system_confidence
        )

        # --------------------------------------------------------
        # Determine decision
        # --------------------------------------------------------

        decision = self._make_decision(
            coverage=coverage,
            mean_confidence=mean_confidence,
            mean_trust=mean_trust,
            conflict_score=conflict_score,
            successful_agents=successful_agents,
            total_agents=total_agents,
            reasoning=reasoning,
        )

        # --------------------------------------------------------
        # Overall status
        # --------------------------------------------------------

        if successful_agents == 0:

            status = "failed"

        elif coverage < 1.0:

            status = "partial"

        else:

            status = "success"

        # --------------------------------------------------------
        # Explanation
        # --------------------------------------------------------

        explanation = self._build_explanation(
            decision=decision,
            coverage=coverage,
            mean_confidence=mean_confidence,
            mean_trust=mean_trust,
            conflict_score=conflict_score,
            clinical_confidence=clinical_confidence,
            successful_agents=successful_agents,
            total_agents=total_agents,
        )

        # --------------------------------------------------------
        # Final output
        # --------------------------------------------------------

        return {
            "status": status,

            "decision": decision,

            "confidence": system_confidence,

            "risk_score": risk_score,

            "coverage": coverage,

            "successful_agents": successful_agents,

            "total_agents": total_agents,

            "mean_confidence": mean_confidence,

            "mean_trust": mean_trust,

            "mean_quality": mean_quality,

            "conflict_score": conflict_score,

            "clinical_reasoning_confidence": (
                clinical_confidence
            ),

            "conflicts": conflicts,

            "clinical_reasoning": reasoning,

            "explanation": explanation,
        }

    # ============================================================
    # DECISION LOGIC
    # ============================================================

    def _make_decision(
        self,
        coverage: float,
        mean_confidence: float,
        mean_trust: float,
        conflict_score: float,
        successful_agents: int,
        total_agents: int,
        reasoning: Dict[str, Any],
    ) -> str:
        """
        Determine system-level decision.

        This deliberately avoids pretending that different
        specialized agents predict one common disease class.
        """

        if successful_agents == 0:

            return "insufficient_data"

        if coverage < self.minimum_coverage:

            return "insufficient_data"

        # --------------------------------------------------------
        # Strong conflict
        # --------------------------------------------------------

        if conflict_score >= 0.75:

            return "requires_review"

        # --------------------------------------------------------
        # Low reliability
        # --------------------------------------------------------

        if mean_trust < self.minimum_trust:

            return "requires_review"

        if mean_confidence < self.minimum_confidence:

            return "requires_review"

        # --------------------------------------------------------
        # Clinical reasoning can influence system action,
        # but not be treated as a disease probability.
        # --------------------------------------------------------

        clinical_prediction = reasoning.get(
            "prediction"
        )

        clinical_confidence = (
            self._clinical_confidence(
                reasoning
            )
        )

        if (
            clinical_prediction is not None
            and clinical_confidence >= 0.75
        ):

            return "clinical_reasoning_supported"

        # --------------------------------------------------------
        # Otherwise sufficient evidence
        # --------------------------------------------------------

        return "evidence_available"

    # ============================================================
    # SYSTEM CONFIDENCE
    # ============================================================

    def _calculate_system_confidence(
        self,
        coverage: float,
        mean_confidence: float,
        mean_trust: float,
        mean_quality: float,
        conflict_score: float,
        clinical_confidence: float,
    ) -> float:

        # --------------------------------------------------------
        # Weighted reliability model
        # --------------------------------------------------------

        base_score = (
            0.25 * coverage
            + 0.25 * mean_confidence
            + 0.25 * mean_trust
            + 0.15 * mean_quality
            + 0.10 * clinical_confidence
        )

        # --------------------------------------------------------
        # Conflict penalty
        # --------------------------------------------------------

        conflict_penalty = (
            0.30 * conflict_score
        )

        score = (
            base_score
            - conflict_penalty
        )

        return self._clip01(
            score
        )

    # ============================================================
    # CONFLICT SCORE
    # ============================================================

    def _calculate_conflict_score(
        self,
        conflicts,
        successful_agents
    ) -> float:

        if not conflicts:
            return 0.0

        # --------------------------------------------------------
        # If conflicts are dictionaries
        # --------------------------------------------------------

        scores = []

        for conflict in conflicts:

            if isinstance(
                conflict,
                dict
            ):

                value = conflict.get(
                    "severity",
                    conflict.get(
                        "score",
                        0.5
                    )
                )

                value = self._safe_float(
                    value,
                    0.5
                )

                # String severity
                if isinstance(
                    conflict.get(
                        "severity"
                    ),
                    str
                ):

                    severity = str(
                        conflict.get(
                            "severity"
                        )
                    ).lower()

                    mapping = {
                        "low": 0.25,
                        "medium": 0.50,
                        "high": 0.75,
                        "critical": 1.00,
                    }

                    value = mapping.get(
                        severity,
                        0.50
                    )

                scores.append(
                    self._clip01(
                        value
                    )
                )

            else:

                # A conflict object exists but doesn't expose
                # a numeric severity.
                scores.append(
                    0.50
                )

        if not scores:
            return 0.0

        return self._clip01(
            sum(scores) / len(scores)
        )

    # ============================================================
    # CLINICAL CONFIDENCE
    # ============================================================

    def _clinical_confidence(
        self,
        reasoning: Dict[str, Any]
    ) -> float:

        if not reasoning:
            return 0.0

        # Direct confidence
        confidence = reasoning.get(
            "confidence"
        )

        if confidence is not None:

            return self._clip01(
                self._safe_float(
                    confidence,
                    0.0
                )
            )

        # Probability
        probability = reasoning.get(
            "probability"
        )

        if probability is not None:

            return self._clip01(
                self._safe_float(
                    probability,
                    0.0
                )
            )

        # --------------------------------------------------------
        # Extract maximum probability from dict
        # --------------------------------------------------------

        probabilities = reasoning.get(
            "probabilities"
        )

        if isinstance(
            probabilities,
            dict
        ):

            values = []

            for value in probabilities.values():

                number = self._safe_float(
                    value
                )

                if number is not None:

                    values.append(
                        self._clip01(
                            number
                        )
                    )

            if values:

                return max(values)

        # --------------------------------------------------------
        # class_probabilities
        # --------------------------------------------------------

        probabilities = reasoning.get(
            "class_probabilities"
        )

        if isinstance(
            probabilities,
            dict
        ):

            values = []

            for value in probabilities.values():

                number = self._safe_float(
                    value
                )

                if number is not None:

                    values.append(
                        self._clip01(
                            number
                        )
                    )

            if values:

                return max(values)

        elif isinstance(
            probabilities,
            (list, tuple)
        ):

            values = []

            for value in probabilities:

                number = self._safe_float(
                    value
                )

                if number is not None:

                    values.append(
                        self._clip01(
                            number
                        )
                    )

            if values:

                return max(values)

        return 0.0

    # ============================================================
    # EXPLANATION
    # ============================================================

    def _build_explanation(
        self,
        decision,
        coverage,
        mean_confidence,
        mean_trust,
        conflict_score,
        clinical_confidence,
        successful_agents,
        total_agents,
    ) -> str:

        parts = []

        parts.append(
            f"{successful_agents}/{total_agents} "
            "agents completed successfully."
        )

        parts.append(
            f"Coverage={coverage:.3f}."
        )

        parts.append(
            f"Mean agent confidence="
            f"{mean_confidence:.3f}."
        )

        parts.append(
            f"Mean agent trust="
            f"{mean_trust:.3f}."
        )

        parts.append(
            f"Conflict score="
            f"{conflict_score:.3f}."
        )

        if clinical_confidence > 0:

            parts.append(
                f"Clinical reasoning confidence="
                f"{clinical_confidence:.3f}."
            )

        parts.append(
            f"System decision: {decision}."
        )

        return " ".join(parts)

    # ============================================================
    # INPUT NORMALIZATION
    # ============================================================

    def _normalize_results(
        self,
        results
    ) -> List[Dict[str, Any]]:

        if results is None:
            return []

        if not isinstance(
            results,
            list
        ):

            results = [results]

        normalized = []

        for result in results:

            if result is None:
                continue

            if isinstance(
                result,
                dict
            ):

                normalized.append(
                    result
                )

            elif hasattr(
                result,
                "to_dict"
            ):

                try:

                    normalized.append(
                        result.to_dict()
                    )

                except Exception:

                    continue

        return normalized

    def _normalize_conflicts(
        self,
        conflicts
    ) -> List[Any]:

        if conflicts is None:
            return []

        if isinstance(
            conflicts,
            list
        ):

            return conflicts

        return [conflicts]

    def _normalize_reasoning(
        self,
        reasoning
    ) -> Dict[str, Any]:

        if reasoning is None:
            return {}

        if isinstance(
            reasoning,
            dict
        ):

            return dict(
                reasoning
            )

        if hasattr(
            reasoning,
            "to_dict"
        ):

            try:

                return reasoning.to_dict()

            except Exception:

                return {}

        return {}

    # ============================================================
    # UTILITY FUNCTIONS
    # ============================================================

    @staticmethod
    def _safe_float(
        value,
        default=None
    ):

        if value is None:
            return default

        try:

            value = float(value)

            if not math.isfinite(value):
                return default

            return value

        except (
            TypeError,
            ValueError
        ):

            return default

    @staticmethod
    def _clip01(
        value
    ):

        try:

            value = float(value)

        except (
            TypeError,
            ValueError
        ):

            return 0.0

        if not math.isfinite(value):
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                value
            )
        )

    @staticmethod
    def _mean(
        values
    ):

        if not values:
            return 0.0

        return sum(
            values
        ) / len(values)
