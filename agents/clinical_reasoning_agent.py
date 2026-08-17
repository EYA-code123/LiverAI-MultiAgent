

# ==========================================================
# Clinical Reasoning Agent
# ==========================================================

class ClinicalReasoningAgent:

    def __init__(self):

        self.name = "ClinicalReasoningAgent"
        self.model_name = "Rule-Based Clinical Synthesis"

    # ======================================================
    # PREDICT
    # ======================================================

    def predict(self, agent_results):

        fatty = agent_results.get("fatty_liver", {})
        fibrosis = agent_results.get("fibrosis", {})
        cirrhosis = agent_results.get("cirrhosis", {})

        results = [fatty, fibrosis, cirrhosis]

        completed = [
            r for r in results
            if r.get("status") == "completed"
        ]

        if len(completed) == 0:

            return {
                "agent": self.name,
                "model": self.model_name,
                "status": "error",
                "error": "No completed agent results"
            }

        # ==================================================
        # CONFIDENCE
        # ==================================================

        probabilities = [
            float(r["probability"])
            for r in completed
            if r.get("probability") is not None
        ]

        if probabilities:
            average_confidence = (
                sum(probabilities) / len(probabilities)
            )
        else:
            average_confidence = 0.0

        # ==================================================
        # CONFIDENCE LEVEL
        # ==================================================

        if average_confidence >= 0.80:
            confidence_level = "High"

        elif average_confidence >= 0.60:
            confidence_level = "Moderate"

        else:
            confidence_level = "Low"

        # ==================================================
        # INTERNAL RISK ASSESSMENT
        # ==================================================

        risk_scores = []

        for result in completed:

            prediction = str(
                result.get("prediction", "")
            )

            probability = float(
                result.get("probability", 0)
            )

            # Prediction codes remain internal.
            # They will NOT be displayed directly in the UI.

            if prediction in ["1", "2", "2.0"]:
                risk_scores.append(probability)

            else:
                risk_scores.append(
                    1 - probability
                )

        # ==================================================
        # OVERALL RISK
        # ==================================================

        if risk_scores:

            overall_score = (
                sum(risk_scores) / len(risk_scores)
            )

        else:

            overall_score = 0.0

        if overall_score >= 0.70:

            overall_risk = "High"

        elif overall_score >= 0.45:

            overall_risk = "Moderate"

        else:

            overall_risk = "Low"

        # ==================================================
        # ABNORMAL FINDINGS
        # ==================================================

        abnormal_findings = []

        if str(fatty.get("prediction")) == "1":

            abnormal_findings.append(
                "Fatty liver assessment requires attention."
            )

        if str(fibrosis.get("prediction")) != "0":

            abnormal_findings.append(
                "Fibrosis assessment requires attention."
            )

        if str(cirrhosis.get("prediction")) in ["1", "2", "2.0"]:

            abnormal_findings.append(
                "Cirrhosis assessment requires attention."
            )

        if not abnormal_findings:

            abnormal_findings = [
                "No major abnormal finding identified by the available agents."
            ]

        # ==================================================
        # CLINICAL DECISION
        # ==================================================

        if overall_risk == "High":

            clinical_decision = (
                "The multi-agent assessment indicates "
                "a high level of concern. Clinical evaluation "
                "and appropriate follow-up are recommended."
            )

        elif overall_risk == "Moderate":

            clinical_decision = (
                "The multi-agent assessment indicates "
                "a moderate level of concern. Clinical "
                "correlation and follow-up are recommended."
            )

        else:

            clinical_decision = (
                "The multi-agent assessment indicates "
                "a relatively low level of concern based "
                "on the available inputs."
            )

        # ==================================================
        # RECOMMENDATION
        # ==================================================

        if overall_risk == "High":

            recommendation = (
                "Consider further clinical assessment "
                "and specialist evaluation."
            )

        elif overall_risk == "Moderate":

            recommendation = (
                "Consider clinical follow-up and, when "
                "appropriate, additional laboratory or "
                "imaging assessment."
            )

        else:

            recommendation = (
                "Continue appropriate clinical monitoring."
            )

        # ==================================================
        # FINAL RESULT
        # ==================================================

        return {

            "agent": self.name,

            "model": self.model_name,

            "status": "completed",

            "agents_used": len(completed),

            "average_confidence": average_confidence,

            "confidence_level": confidence_level,

            "overall_risk": overall_risk,

            "risk_score": overall_score,

            "abnormal_findings": abnormal_findings,

            "clinical_decision": clinical_decision,

            "recommendation": recommendation,

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

            "summary": (
                "The three specialized liver agents were "
                "successfully integrated by the clinical "
                "reasoning agent."
            )
        }
