%%writefile /content/LiverAI-MultiAgent/agents/clinical_reasoning_agent.py

class ClinicalReasoningAgent:

    def __init__(self):

        self.name = "ClinicalReasoningAgent"
        self.model_name = "Rule-Based Synthesis"

    def predict(self, agent_results):

        fatty = agent_results.get(
            "fatty_liver",
            {}
        )

        fibrosis = agent_results.get(
            "fibrosis",
            {}
        )

        cirrhosis = agent_results.get(
            "cirrhosis",
            {}
        )

        results = [
            fatty,
            fibrosis,
            cirrhosis
        ]

        completed = [
            r for r in results
            if r.get("status") == "completed"
        ]

        if len(completed) == 0:

            return {
                "agent": self.name,
                "model": self.model_name,
                "status": "error",
                "error": "No agent results available"
            }

        # ==================================================
        # COLLECT CONFIDENCE
        # ==================================================

        probabilities = []

        for result in completed:

            probability = result.get(
                "probability"
            )

            if probability is not None:

                probabilities.append(
                    float(probability)
                )

        # ==================================================
        # GLOBAL CONFIDENCE
        # ==================================================

        if len(probabilities) > 0:

            average_confidence = (
                sum(probabilities)
                / len(probabilities)
            )

        else:

            average_confidence = None

        # ==================================================
        # RISK SUMMARY
        # ==================================================

        if average_confidence is None:

            risk_summary = "UNDETERMINED"

        elif average_confidence >= 0.80:

            risk_summary = "HIGH_CONFIDENCE"

        elif average_confidence >= 0.60:

            risk_summary = "MODERATE_CONFIDENCE"

        else:

            risk_summary = "LOW_CONFIDENCE"

        # ==================================================
        # IMPORTANT:
        # DO NOT INTERPRET MODEL LABELS AS DIAGNOSIS
        # ==================================================

        clinical_summary = (
            "The multi-agent system successfully integrated "
            "the available model outputs. The predictions "
            "are presented as model outputs and should not "
            "be interpreted as a clinical diagnosis."
        )

        # ==================================================
        # FINAL RESULT
        # ==================================================

        return {

            "agent": self.name,

            "model": self.model_name,

            "status": "completed",

            "agents_used": len(completed),

            "fatty_liver": {
                "prediction": fatty.get(
                    "prediction"
                ),
                "probability": fatty.get(
                    "probability"
                )
            },

            "fibrosis": {
                "prediction": fibrosis.get(
                    "prediction"
                ),
                "probability": fibrosis.get(
                    "probability"
                )
            },

            "cirrhosis": {
                "prediction": cirrhosis.get(
                    "prediction"
                ),
                "probability": cirrhosis.get(
                    "probability"
                )
            },

            "average_confidence": (
                average_confidence
            ),

            "risk_summary": risk_summary,

            "clinical_summary": clinical_summary
        }
