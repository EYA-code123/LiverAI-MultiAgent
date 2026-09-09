# =============================================================================
# LiverAI-MultiAgent
# ACTION ENGINE
# =============================================================================

from typing import Any, Dict


class ActionEngine:
    """
    Converts task-specific decisions into explicit coordination actions.

    Supported coordination actions:

        ACCEPT
        QUERY
        REASSESS
        REQUEST_DATA
        ESCALATE

    The engine never performs a clinical intervention itself.
    It only produces a structured recommendation for the next
    coordination step.
    """

    ACCEPT = "ACCEPT"

    QUERY = "QUERY"

    REASSESS = "REASSESS"

    REQUEST_DATA = "REQUEST_DATA"

    ESCALATE = "ESCALATE"

    def __init__(self):
        pass

    # =========================================================================
    # MAIN ENTRY
    # =========================================================================

    def generate(
        self,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:

        if not isinstance(
            decision,
            dict
        ):

            return {
                "status": "error",
                "task_actions": {},
                "error": (
                    "Decision must be a dictionary."
                ),
            }

        task_decisions = decision.get(
            "task_decisions",
            {}
        )

        if not isinstance(
            task_decisions,
            dict
        ):

            task_decisions = {}

        task_actions = {}

        for task_type, task_decision in (
            task_decisions.items()
        ):

            if not isinstance(
                task_decision,
                dict
            ):

                continue

            task_actions[
                task_type
            ] = self._generate_task_action(
                task_type,
                task_decision
            )

        return {
            "status": "completed",
            "task_actions": task_actions,
        }

    # =========================================================================
    # TASK ACTION
    # =========================================================================

    def _generate_task_action(
        self,
        task_type: str,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:

        decision_level = str(
            decision.get(
                "decision_level",
                decision.get(
                    "decision",
                    "UNCERTAIN"
                )
            )
        ).upper()

        confidence = self._safe_float(
            decision.get(
                "confidence",
                0.0
            )
        )

        risk_score = self._safe_float(
            decision.get(
                "risk_score",
                1.0
            )
        )

        prediction = decision.get(
            "prediction"
        )

        additional_tests = bool(
            decision.get(
                "request_additional_tests",
                False
            )
        )

        # =====================================================================
        # SEGMENTATION
        # =====================================================================

        if task_type == "liver_segmentation":

            return self._segmentation_action(
                decision_level=decision_level,
                confidence=confidence,
                risk_score=risk_score,
                additional_tests=additional_tests,
            )

        # =====================================================================
        # HIGH
        # =====================================================================

        if decision_level == "HIGH":

            return {

                "status":
                    "high_confidence",

                "coordination_action":
                    self.ACCEPT,

                "actions": [

                    (
                        "Finding requiring "
                        f"clinical validation: "
                        f"{prediction}."
                    ),

                    "Review the supporting evidence.",

                    (
                        "Consider specialist "
                        "confirmation before "
                        "clinical intervention."
                    ),
                ],

                "referral":
                    True,

                "follow_up":
                    True,

                "additional_tests":
                    additional_tests,

                "risk_score":
                    risk_score,

                "confidence":
                    confidence,
            }

        # =====================================================================
        # MODERATE
        # =====================================================================

        if decision_level == "MODERATE":

            return {

                "status":
                    "moderate_confidence",

                "coordination_action":
                    self.QUERY,

                "actions": [

                    (
                        "Preliminary finding: "
                        f"{prediction}."
                    ),

                    "Perform clinical review.",

                    (
                        "Consider additional "
                        "evidence if clinically "
                        "indicated."
                    ),
                ],

                "referral":
                    True,

                "follow_up":
                    True,

                "additional_tests":
                    True,

                "risk_score":
                    risk_score,

                "confidence":
                    confidence,
            }

        # =====================================================================
        # UNCERTAIN
        # =====================================================================

        return {

            "status":
                "cautious",

            "coordination_action":
                self.ESCALATE,

            "actions": [

                (
                    "Additional clinical "
                    "assessment is recommended."
                ),

                (
                    "Consider additional "
                    "imaging or laboratory data."
                ),

                (
                    "Specialist review is "
                    "recommended."
                ),

                (
                    "Do not use this automated "
                    "output as a standalone "
                    "diagnosis."
                ),
            ],

            "referral":
                True,

            "follow_up":
                True,

            "additional_tests":
                True,

            "risk_score":
                risk_score,

            "confidence":
                confidence,
        }

    # =========================================================================
    # SEGMENTATION ACTION
    # =========================================================================

    def _segmentation_action(
        self,
        decision_level,
        confidence,
        risk_score,
        additional_tests,
    ):

        if decision_level == "HIGH":

            return {

                "status":
                    "high_confidence",

                "coordination_action":
                    self.ACCEPT,

                "actions": [

                    (
                        "Liver segmentation "
                        "completed successfully."
                    ),

                    (
                        "Review the generated "
                        "liver mask and "
                        "probability map."
                    ),

                    (
                        "Use the segmentation "
                        "as imaging support "
                        "for further analysis."
                    ),

                    (
                        "Do not use automated "
                        "segmentation as a "
                        "standalone diagnosis."
                    ),
                ],

                "referral":
                    True,

                "follow_up":
                    True,

                "additional_tests":
                    additional_tests,

                "risk_score":
                    risk_score,

                "confidence":
                    confidence,
            }

        if decision_level == "MODERATE":

            return {

                "status":
                    "moderate_confidence",

                "coordination_action":
                    self.REASSESS,

                "actions": [

                    (
                        "Liver segmentation "
                        "completed with "
                        "moderate confidence."
                    ),

                    (
                        "Review the generated "
                        "liver mask and "
                        "probability map."
                    ),

                    (
                        "Consider additional "
                        "imaging evidence."
                    ),

                    (
                        "Specialist review "
                        "is recommended."
                    ),
                ],

                "referral":
                    True,

                "follow_up":
                    True,

                "additional_tests":
                    True,

                "risk_score":
                    risk_score,

                "confidence":
                    confidence,
            }

        return {

            "status":
                "cautious",

            "coordination_action":
                self.REQUEST_DATA,

            "actions": [

                (
                    "Liver segmentation "
                    "evidence is insufficient."
                ),

                (
                    "Review the segmentation "
                    "output if available."
                ),

                (
                    "Consider additional "
                    "imaging evidence."
                ),

                (
                    "Specialist review "
                    "is recommended."
                ),
            ],

            "referral":
                True,

            "follow_up":
                True,

            "additional_tests":
                True,

            "risk_score":
                risk_score,

            "confidence":
                confidence,
        }

    # =========================================================================
    # SAFE FLOAT
    # =========================================================================

    @staticmethod
    def _safe_float(
        value,
        default=0.0
    ):

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            value = default

        return max(
            0.0,
            min(
                1.0,
                value
            )
        )


__all__ = [
    "ActionEngine",
]
