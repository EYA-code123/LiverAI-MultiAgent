%%writefile /content/LiverAI-MultiAgent/agents/clinical_reasoning_agent.py

class ClinicalReasoningAgent:

    def __init__(self):

        self.name = "ClinicalReasoningAgent"
        self.model_name = "Rule-Based Synthesis"

    def predict(self, agent_results):

        fatty = agent_results.get("fatty_liver", {})
        fibrosis = agent_results.get("fibrosis", {})
        cirrhosis = agent_results.get("cirrhosis", {})

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
        # CONFIDENCE VALUES
        # ==================================================

        probabilities = []

        for result in completed:

            probability = result.get("probability")

            if probability is not None:
                probabilities.append(
                    float(probability)
                )

        # ==================================================
        # AVERAGE CONFIDENCE
        # ==================================================

        if probabilities:

            average_confidence = (
                sum(probabilities)
                / len(probabilities)
            )

        else:

            average_confidence = None

        # ==================================================
        # CONFIDENCE CATEGORY
        # ==================================================

        if average_confidence is None:

            confidence_level = "UNDETERMINED"

        elif average_confidence >= 0.80:

            confidence_level = "HIGH"

        elif average_confidence >= 0.60:

            confidence_level = "MODERATE"

        else:

            confidence_level = "LOW"

        # ==================================================
        # CLINICAL SAFETY
        # ==================================================

        clinical_summary = (
            "The system successfully integrated the outputs "
            "of the three liver assessment models. "
            "These outputs are model predictions and confidence "
            "scores and must not be interpreted as a clinical "
            "diagnosis."
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
                "prediction": fatty.get("prediction"),
                "probability": fatty.get("probability")
            },

            "fibrosis": {
                "prediction": fibrosis.get("prediction"),
                "probability": fibrosis.get("probability")
            },

            "cirrhosis": {
                "prediction": cirrhosis.get("prediction"),
                "probability": cirrhosis.get("probability")
            },

            "average_confidence": average_confidence,

            "confidence_level": confidence_level,

            "clinical_summary": clinical_summary
        }
