"""
Action Intelligence
====================

Generates explainable next-step recommendations from
the confidence-aware decision layer.

This is decision support, not autonomous medical treatment.
"""


class ActionIntelligence:

    def generate(
        self,
        decisions,
        reasoning
    ):

        recommendations = []

        task_decisions = decisions.get(
            "task_decisions",
            {}
        )

        for task, decision in task_decisions.items():

            confidence = decision.get(
                "confidence",
                0.0
            )

            request_tests = decision.get(
                "request_additional_tests",
                False
            )

            if request_tests:

                recommendations.append({
                    "task": task,
                    "action":
                        "Request additional clinical "
                        "or imaging evidence before "
                        "making a high-confidence decision.",
                    "priority": "high"
                })

            elif confidence >= 0.80:

                recommendations.append({
                    "task": task,
                    "action":
                        "Prediction is supported by "
                        "high-confidence model evidence.",
                    "priority": "normal"
                })

            else:

                recommendations.append({
                    "task": task,
                    "action":
                        "Review the prediction together "
                        "with the available clinical evidence.",
                    "priority": "moderate"
                })

        return {
            "recommendations":
                recommendations,

            "explainable_summary":
                reasoning.get(
                    "reasoning",
                    ""
                )
        }
