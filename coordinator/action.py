# ============================================================
# LIVERAI - ACTION INTELLIGENCE
# ============================================================


class ActionEngine:

    def __init__(self):
        pass

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(self, decision):

        task_decisions = decision.get(
            "task_decisions",
            {}
        )

        task_actions = {}

        for task_type, task_decision in task_decisions.items():

            decision_level = task_decision.get(
                "decision_level",
                task_decision.get(
                    "decision",
                    "UNCERTAIN"
                )
            )

            decision_level = str(
                decision_level
            ).upper()

            confidence = float(
                task_decision.get(
                    "confidence",
                    0.0
                ) or 0.0
            )

            risk_score = float(
                task_decision.get(
                    "risk_score",
                    1.0
                ) or 1.0
            )

            additional_tests = bool(
                task_decision.get(
                    "request_additional_tests",
                    False
                )
            )

            prediction = task_decision.get(
                "prediction"
            )

            # =================================================
            # SEGMENTATION
            # =================================================

            if task_type == "liver_segmentation":

                if decision_level == "HIGH":

                    action = {
                        "status": "high_confidence",
                        "coordination_action": "ACCEPT",
                        "actions": [
                            "Liver segmentation completed successfully.",
                            "Review the generated liver mask and probability map.",
                            "Use the segmentation as imaging support for further analysis.",
                            "Do not use automated segmentation as a standalone diagnosis.",
                        ],
                        "referral": True,
                        "follow_up": True,
                        "additional_tests": additional_tests,
                        "risk_score": risk_score,
                        "confidence": confidence,
                    }

                elif decision_level == "MODERATE":

                    action = {
                        "status": "moderate_confidence",
                        "coordination_action": "REASSESS",
                        "actions": [
                            "Liver segmentation completed with moderate confidence.",
                            "Review the generated liver mask and probability map.",
                            "Consider additional imaging evidence.",
                            "Specialist review is recommended.",
                        ],
                        "referral": True,
                        "follow_up": True,
                        "additional_tests": True,
                        "risk_score": risk_score,
                        "confidence": confidence,
                    }

                else:

                    action = {
                        "status": "cautious",
                        "coordination_action": "REQUEST_DATA",
                        "actions": [
                            "Liver segmentation evidence is insufficient.",
                            "Review the segmentation output if available.",
                            "Consider additional imaging evidence.",
                            "Specialist review is recommended.",
                        ],
                        "referral": True,
                        "follow_up": True,
                        "additional_tests": True,
                        "risk_score": risk_score,
                        "confidence": confidence,
                    }

                task_actions[task_type] = action

                continue

            # =================================================
            # HIGH CONFIDENCE
            # =================================================

            if decision_level == "HIGH":

                task_actions[task_type] = {
                    "status": "high_confidence",
                    "coordination_action": "ACCEPT",
                    "actions": [
                        f"Finding requiring clinical validation: {prediction}.",
                        "Review the supporting evidence.",
                        "Consider specialist confirmation before intervention.",
                    ],
                    "referral": True,
                    "follow_up": True,
                    "additional_tests": additional_tests,
                    "risk_score": risk_score,
                    "confidence": confidence,
                }

            # =================================================
            # MODERATE
            # =================================================

            elif decision_level == "MODERATE":

                task_actions[task_type] = {
                    "status": "moderate_confidence",
                    "coordination_action": "QUERY",
                    "actions": [
                        f"Preliminary finding: {prediction}.",
                        "Perform clinical review.",
                        "Consider additional evidence if clinically indicated.",
                    ],
                    "referral": True,
                    "follow_up": True,
                    "additional_tests": True,
                    "risk_score": risk_score,
                    "confidence": confidence,
                }

            # =================================================
            # UNCERTAIN
            # =================================================

            else:

                task_actions[task_type] = {
                    "status": "cautious",
                    "coordination_action": "ESCALATE",
                    "actions": [
                        "Additional clinical assessment is recommended.",
                        "Consider additional imaging or laboratory data.",
                        "Specialist review is recommended.",
                        "Do not use this automated output as a standalone diagnosis.",
                    ],
                    "referral": True,
                    "follow_up": True,
                    "additional_tests": True,
                    "risk_score": risk_score,
                    "confidence": confidence,
                }

        # =====================================================
        # IMPORTANT: TOP-LEVEL STATUS
        # =====================================================

        return {
            "status": "completed",
            "task_actions": task_actions,
        }
