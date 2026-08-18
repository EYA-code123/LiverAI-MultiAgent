

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
    # INTERPRETATION
    # ======================================================

    def _interpret_fatty_liver(self, result):

        if result.get("status") != "completed":
            return "Assessment unavailable"

        prediction = str(
            result.get("prediction")
        )

        if prediction == "1":
            return "Fatty liver pattern detected"

        return "No fatty liver pattern detected"


    def _interpret_fibrosis(self, result):

        if result.get("status") != "completed":
            return "Assessment unavailable"

        prediction = str(
            result.get("prediction")
        )

        if prediction == "1":
            return "Fibrosis pattern detected"

        return "No fibrosis pattern detected"


    def _interpret_cirrhosis(self, result):

        if result.get("status") != "completed":
            return "Assessment unavailable"

        prediction = str(
            result.get("prediction")
        )

        if prediction == "0":

            return (
                "Lower-risk cirrhosis category"
            )

        elif prediction == "1":

            return (
                "Intermediate cirrhosis category"
            )

        elif prediction == "2.0":

            return (
                "Advanced cirrhosis category"
            )

        return "Cirrhosis category detected"


    # ======================================================
    # PREDICTION
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

        completed = [
            result
            for result in results
            if result.get("status") == "completed"
        ]

        # --------------------------------------------------
        # No completed agents
        # --------------------------------------------------

        if len(completed) == 0:

            return {

                "agent":
                    self.name,

                "model":
                    self.model_name,

                "status":
                    "error",

                "error":
                    "No completed agent results available"
            }


        # --------------------------------------------------
        # Confidence
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


        if probabilities:

            average_confidence = (
                sum(probabilities)
                / len(probabilities)
            )

        else:

            average_confidence = 0.0


        # --------------------------------------------------
        # Risk calculation
        # --------------------------------------------------

        abnormal_findings = 0

        if str(
            fatty.get("prediction")
        ) == "1":

            abnormal_findings += 1


        if str(
            fibrosis.get("prediction")
        ) == "1":

            abnormal_findings += 1


        if str(
            cirrhosis.get("prediction")
        ) == "2.0":

            abnormal_findings += 1


        if abnormal_findings >= 2:

            overall_risk = "High"

        elif abnormal_findings == 1:

            overall_risk = "Moderate"

        else:

            overall_risk = "Low"


        # --------------------------------------------------
        # Confidence level
        # --------------------------------------------------

        if average_confidence >= 0.80:

            confidence_level = "High"

        elif average_confidence >= 0.60:

            confidence_level = "Moderate"

        else:

            confidence_level = "Low"


        # --------------------------------------------------
        # Clinical decision
        # --------------------------------------------------

        if overall_risk == "High":

            clinical_decision = (
                "Multiple liver-related abnormalities "
                "were identified. Further clinical "
                "evaluation is recommended."
            )

        elif overall_risk == "Moderate":

            clinical_decision = (
                "The multi-agent assessment identified "
                "a potential liver abnormality. Clinical "
                "correlation and follow-up are recommended."
            )

        else:

            clinical_decision = (
                "No major abnormal pattern was identified "
                "by the available assessment agents. "
                "Clinical correlation remains necessary."
            )


        # --------------------------------------------------
        # Agent findings
        # --------------------------------------------------

        findings = {

            "fatty_liver": {

                "assessment":
                    self._interpret_fatty_liver(
                        fatty
                    ),

                "confidence":
                    fatty.get("probability")
            },


            "fibrosis": {

                "assessment":
                    self._interpret_fibrosis(
                        fibrosis
                    ),

                "confidence":
                    fibrosis.get("probability")
            },


            "cirrhosis": {

                "assessment":
                    self._interpret_cirrhosis(
                        cirrhosis
                    ),

                "confidence":
                    cirrhosis.get("probability")
            }
        }


        # --------------------------------------------------
        # Recommendation
        # --------------------------------------------------

        recommendation = (
            "This AI-generated assessment is intended "
            "for clinical decision support only and "
            "should be interpreted by a qualified "
            "healthcare professional."
        )


        # --------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------

        return {

            "agent":
                self.name,

            "model":
                self.model_name,

            "status":
                "completed",

            "agents_used":
                len(completed),

            "average_confidence":
                average_confidence,

            "overall_risk":
                overall_risk,

            "confidence_level":
                confidence_level,

            "abnormal_findings":
                abnormal_findings,

            "findings":
                findings,

            "clinical_decision":
                clinical_decision,

            "recommendation":
                recommendation
        }
