from pathlib import Path

action_code = r'''
class ActionEngine:

    def __init__(self):
        pass

    def generate(self, decision):
        """
        Generate task-specific actions from task-level decisions.
        """

        task_decisions = decision.get("task_decisions", {})

        task_actions = {}

        for task_type, task_decision in task_decisions.items():

            decision_level = task_decision.get(
                "decision_level",
                task_decision.get("decision", "UNCERTAIN")
            )

            confidence = float(
                task_decision.get("confidence", 0.0) or 0.0
            )

            risk_score = float(
                task_decision.get("risk_score", 1.0) or 1.0
            )

            additional_tests = bool(
                task_decision.get("request_additional_tests", False)
            )

            # =========================================================
            # LIVER SEGMENTATION
            # =========================================================

            if task_type == "liver_segmentation":

                if decision_level == "HIGH":
                    task_actions[task_type] = {
                        "status": "high_confidence",
                        "actions": [
                            "Liver segmentation completed successfully.",
                            "Review the generated liver mask and probability map.",
                            "Use the segmentation as imaging support for further analysis.",
                            "Do not use the automated segmentation as a standalone diagnosis."
                        ],
                        "referral": True,
                        "follow_up": True,
                        "additional_tests": additional_tests,
                        "risk_score": risk_score,
                        "confidence": confidence
                    }

                elif decision_level == "MODERATE":
                    task_actions[task_type] = {
                        "status": "moderate_confidence",
                        "actions": [
                            "Liver segmentation completed with moderate confidence.",
                            "Review the generated liver mask and probability map.",
                            "Consider additional imaging evidence if clinically indicated.",
                            "Specialist review is recommended before clinical use."
                        ],
                        "referral": True,
                        "follow_up": True,
                        "additional_tests": True,
                        "risk_score": risk_score,
                        "confidence": confidence
                    }

                else:
                    task_actions[task_type] = {
                        "status": "cautious",
                        "actions": [
                            "Liver segmentation evidence is insufficient.",
                            "Review the generated segmentation output if available.",
                            "Consider additional imaging or clinical evidence.",
                            "Specialist review recommended.",
                            "Do not use this automated output as a standalone diagnosis."
                        ],
                        "referral": True,
                        "follow_up": True,
                        "additional_tests": True,
                        "risk_score": risk_score,
                        "confidence": confidence
                    }

                continue

            # =========================================================
            # CLASSIFICATION / OTHER TASKS
            # =========================================================

            prediction = task_decision.get("prediction")

            if decision_level == "HIGH":
                task_actions[task_type] = {
                    "status": "high_confidence",
                    "actions": [
                        f"Finding requiring clinical validation: {prediction}.",
                        "Review the supporting evidence.",
                        "Consider specialist confirmation before intervention."
                    ],
                    "referral": True,
                    "follow_up": True,
                    "additional_tests": additional_tests,
                    "risk_score": risk_score,
                    "confidence": confidence
                }

            elif decision_level == "MODERATE":
                task_actions[task_type] = {
                    "status": "moderate_confidence",
                    "actions": [
                        f"Preliminary finding: {prediction}.",
                        "Perform clinical review.",
                        "Consider additional evidence if clinically indicated."
                    ],
                    "referral": True,
                    "follow_up": True,
                    "additional_tests": True,
                    "risk_score": risk_score,
                    "confidence": confidence
                }

            else:
                task_actions[task_type] = {
                    "status": "cautious",
                    "actions": [
                        "Additional clinical assessment recommended.",
                        "Consider additional imaging or laboratory data.",
                        "Specialist review recommended.",
                        "Do not use this automated output as a standalone diagnosis."
                    ],
                    "referral": True,
                    "follow_up": True,
                    "additional_tests": True,
                    "risk_score": risk_score,
                    "confidence": confidence
                }

        return {
            "task_actions": task_actions
        }
'''

path = Path("/content/LiverAI-MultiAgent/coordinator/action.py")
path.write_text(action_code, encoding="utf-8")

print("✅ coordinator/action.py replaced")
print("Path:", path)
print("Size:", path.stat().st_size, "bytes")
