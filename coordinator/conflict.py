"""
Decision Intelligence
======================

Confidence-aware final clinical decision layer.
"""

import numpy as np


class DecisionEngine:

    def __init__(
        self,
        minimum_confidence=0.60,
        high_confidence=0.80
    ):

        self.minimum_confidence = (
            minimum_confidence
        )

        self.high_confidence = (
            high_confidence
        )

    @staticmethod
    def _clip(value):

        return float(
            np.clip(
                float(value),
                0.0,
                1.0
            )
        )

    def _risk_level(
        self,
        confidence,
        uncertainty
    ):

        confidence = self._clip(
            confidence
        )

        uncertainty = self._clip(
            uncertainty
        )

        risk = (
            0.60 * uncertainty
            +
            0.40 * (1.0 - confidence)
        )

        if risk >= 0.65:
            return "high"

        if risk >= 0.35:
            return "moderate"

        return "low"

    def decide(
        self,
        agent_results,
        conflicts=None,
        fused_results=None
    ):

        conflicts = conflicts or []

        fused_results = (
            fused_results
            or {}
        )

        successful_agents = [
            result
            for result in agent_results
            if getattr(
                result,
                "status",
                "success"
            ) == "success"
        ]

        failed_agents = [
            result
            for result in agent_results
            if getattr(
                result,
                "status",
                "success"
            ) != "success"
        ]

        task_decisions = {}

        for task_type, result in fused_results.items():

            if result is None:
                continue

            confidence = self._clip(
                result.get(
                    "confidence",
                    0.0
                )
            )

            uncertainty = self._clip(
                result.get(
                    "uncertainty",
                    1.0 - confidence
                )
            )

            prediction = result.get(
                "prediction"
            )

            if confidence >= self.high_confidence:

                decision_level = "high_confidence"

            elif confidence >= self.minimum_confidence:

                decision_level = "moderate_confidence"

            else:

                decision_level = "low_confidence"

            request_additional_tests = (
                confidence
                < self.minimum_confidence
                or uncertainty > 0.50
            )

            task_decisions[task_type] = {
                "prediction": prediction,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "risk_level": self._risk_level(
                    confidence,
                    uncertainty
                ),
                "decision_level": decision_level,
                "request_additional_tests":
                    request_additional_tests
            }

        overall_confidences = [
            item["confidence"]
            for item in task_decisions.values()
        ]

        overall_confidence = (
            float(np.mean(overall_confidences))
            if overall_confidences
            else 0.0
        )

        return {
            "status": (
                "completed"
                if successful_agents
                else "failed"
            ),

            "num_agents": len(
                agent_results
            ),

            "successful_agents": len(
                successful_agents
            ),

            "failed_agents": len(
                failed_agents
            ),

            "num_conflicts": len(
                conflicts
            ),

            "overall_confidence":
                overall_confidence,

            "request_additional_tests":
                overall_confidence
                < self.minimum_confidence,

            "task_decisions":
                task_decisions,

            "warning": (
                "Conflicting agent predictions detected."
                if conflicts
                else None
            )
        }
