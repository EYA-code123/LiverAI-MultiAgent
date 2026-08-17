
# ==========================================================
# LiverAI - Clinical Reasoning Agent
# ==========================================================

class ClinicalReasoningAgent:

    def __init__(self):

        self.name = "ClinicalReasoningAgent"
        self.model_name = "Rule-Based Clinical Synthesis"

    # ======================================================
    # PREDICT / SYNTHESIS
    # ======================================================

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

        # --------------------------------------------------
        # Check completed agents
        # --------------------------------------------------

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
                "error": "No valid agent results available"
            }

        # --------------------------------------------------
        # Extract probabilities
        # --------------------------------------------------

        probabilities = []

        for result in completed:

            probability = result.get(
                "probability"
            )

            if probability is not None:

                probabilities.append(
                    float(probability)
                )

        # --------------------------------------------------
        # Average confidence
        # --------------------------------------------------

        if len(probabilities) > 0:

            average_confidence = (
                sum(probabilities)
                / len(probabilities)
            )

        else:

            average_confidence = 0.0

        # --------------------------------------------------
        # Determine risk level
        # --------------------------------------------------

        if average_confidence >= 0.80:

            risk_level = "High confidence"

        elif average_confidence >= 0.60:

            risk_level = "Moderate confidence"

        else:

            risk_level = "Low confidence"

        # --------------------------------------------------
        # Agent availability
        # --------------------------------------------------

        agent_status = {

            "fatty_liver":
                fatty.get("status") == "completed",

            "fibrosis":
                fibrosis.get("status") == "completed",

            "cirrhosis":
                cirrhosis.get("status") == "completed"
        }

        # --------------------------------------------------
        # Clinical summary
        # --------------------------------------------------

        summary = (
            f"{len(completed)} liver assessment agents "
            f"successfully contributed to the clinical synthesis."
        )

        # --------------------------------------------------
        # Recommendation
        # --------------------------------------------------

        recommendation = (
            "The results should be interpreted together "
            "with clinical history, laboratory findings, "
            "imaging and evaluation by a qualified healthcare professional."
        )

        # --------------------------------------------------
        # Final result
        # --------------------------------------------------

        return {

            "agent": self.name,

            "model": self.model_name,

            "status": "completed",

            "agents_used": len(completed),

            "average_confidence":
                round(
                    average_confidence,
                    4
                ),

            "risk_level":
                risk_level,

            "agent_status":
                agent_status,

            "fatty_liver": {

                "prediction":
                    fatty.get("prediction"),

                "probability":
                    fatty.get("probability")
            },

            "fibrosis": {

                "prediction":
                    fibrosis.get("prediction"),

                "probability":
                    fibrosis.get("probability")
            },

            "cirrhosis": {

                "prediction":
                    cirrhosis.get("prediction"),

                "probability":
                    cirrhosis.get("probability")
            },

            "clinical_summary":
                summary,

            "recommendation":
                recommendation
        }
