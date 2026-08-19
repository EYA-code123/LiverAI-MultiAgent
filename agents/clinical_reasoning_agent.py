%%writefile /content/LiverAI-MultiAgent/agents/clinical_reasoning_agent.py

class ClinicalReasoningAgent:

    def __init__(self):

        self.name = "ClinicalReasoningAgent"
        self.model_name = "Rule-Based Clinical Reasoning"

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

        # ---------------------------------------------
        # Extract predictions
        # ---------------------------------------------

        fatty_prediction = fatty.get(
            "prediction"
        )

        fibrosis_prediction = fibrosis.get(
            "prediction"
        )

        cirrhosis_prediction = cirrhosis.get(
            "prediction"
        )

        # ---------------------------------------------
        # Extract probabilities
        # ---------------------------------------------

        fatty_probability = fatty.get(
            "probability"
        )

        fibrosis_probability = fibrosis.get(
            "probability"
        )

        cirrhosis_probability = cirrhosis.get(
            "probability"
        )

        # ---------------------------------------------
        # Findings
        # ---------------------------------------------

        findings = []

        findings.append(
            f"Fatty liver prediction: {fatty_prediction}"
        )

        findings.append(
            f"Fibrosis prediction: {fibrosis_prediction}"
        )

        findings.append(
            f"Cirrhosis prediction: {cirrhosis_prediction}"
        )

        # ---------------------------------------------
        # Overall risk
        # ---------------------------------------------

        if str(cirrhosis_prediction) in ["1", "2"]:

            overall_risk = "Elevated"

        elif str(fibrosis_prediction) == "1":

            overall_risk = "Moderate"

        elif str(fatty_prediction) in ["1", "2"]:

            overall_risk = "Possible fatty liver"

        else:

            overall_risk = "Low"

        # ---------------------------------------------
        # Result
        # ---------------------------------------------

        return {

            "agent": self.name,

            "model": self.model_name,

            "overall_risk": overall_risk,

            "findings": findings,

            "fatty_liver_probability":
                fatty_probability,

            "fibrosis_probability":
                fibrosis_probability,

            "cirrhosis_probability":
                cirrhosis_probability,

            "status": "completed"
        }
