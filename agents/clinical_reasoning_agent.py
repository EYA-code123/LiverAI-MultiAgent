%%writefile /content/LiverAI-MultiAgent/agents/clinical_reasoning_agent.py

# ==========================================================
# LiverAI - Clinical Reasoning Agent
# ==========================================================

class ClinicalReasoningAgent:

    def __init__(self):

        self.name = "ClinicalReasoningAgent"

        self.model_name = (
            "Rule-Based Clinical Synthesis"
        )

    # ======================================================
    # PREDICT
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

        results = [
            fatty,
            fibrosis,
            cirrhosis
        ]

        # --------------------------------------------------
        # Completed agents
        # --------------------------------------------------

        completed = [
            r for r in results
            if r.get("status") == "completed"
        ]

        if not completed:

            return {
                "agent": self.name,
                "model": self.model_name,
                "status": "error",
                "error": "No completed agent results"
            }

        # ==================================================
        # CONFIDENCE
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

        if probabilities:

            average_confidence = (
                sum(probabilities)
                / len(probabilities)
            )

        else:

            average_confidence = 0.0

        # --------------------------------------------------
        # Confidence level
        # --------------------------------------------------

        if average_confidence >= 0.80:

            confidence_level = "High"

        elif average_confidence >= 0.60:

            confidence_level = "Moderate"

        else:

            confidence_level = "Low"

        # ==================================================
        # RISK SCORE
        # ==================================================

        risk_scores = []

        for result in completed:

            prediction = str(
                result.get(
                    "prediction",
                    ""
                )
            )

            probability = float(
                result.get(
                    "probability",
                    0.0
                )
            )

            # Abnormal classes
            if prediction in [
                "1",
                "2",
                "2.0"
            ]:

                risk_scores.append(
                    probability
                )

            # Normal class
            else:

                risk_scores.append(
                    1.0 - probability
                )

        if risk_scores:

            overall_score = (
                sum(risk_scores)
                / len(risk_scores)
            )

        else:

            overall_score = 0.0

        # --------------------------------------------------
        # Overall risk
        # --------------------------------------------------

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

        # Fatty liver
        if str(
            fatty.get("prediction")
        ) == "1":

            abnormal_findings.append(
                "Fatty liver assessment requires attention."
            )

        # Fibrosis
        if str(
            fibrosis.get("prediction")
        ) != "0":

            abnormal_findings.append(
                "Fibrosis assessment requires attention."
            )

        # Cirrhosis
        if str(
            cirrhosis.get("prediction")
        ) in [
            "1",
            "2",
            "2.0"
        ]:

            abnormal_findings.append(
                "Cirrhosis assessment requires attention."
            )

        if not abnormal_findings:

            abnormal_findings.append(
                "No major abnormal finding identified "
                "by the available agents."
            )

        # ==================================================
        # CLINICAL DECISION
        # ==================================================

        if overall_risk == "High":

            clinical_decision = (
                "The multi-agent assessment indicates "
                "a high level of concern. Clinical "
                "evaluation and appropriate follow-up "
                "are recommended."
            )

            recommendation = (
                "Consider further clinical assessment "
                "and specialist evaluation."
            )

        elif overall_risk == "Moderate":

            clinical_decision = (
                "The multi-agent assessment indicates "
                "a moderate level of concern. Clinical "
                "correlation and follow-up are recommended."
            )

            recommendation = (
                "Consider clinical follow-up and, when "
                "appropriate, additional laboratory or "
                "imaging assessment."
            )

        else:

            clinical_decision = (
                "The multi-agent assessment indicates "
                "a relatively low level of concern "
                "based on the available inputs."
            )

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

            "average_confidence":
                average_confidence,

            "confidence_level":
                confidence_level,

            "overall_risk":
                overall_risk,

            "risk_score":
                overall_score,

            "abnormal_findings":
                abnormal_findings,

            "clinical_decision":
                clinical_decision,

            "recommendation":
                recommendation,

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

            "summary": (
                "The three specialized liver agents "
                "were successfully integrated by "
                "the Clinical Reasoning Agent."
            )
        }
