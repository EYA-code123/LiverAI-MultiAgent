class ClinicalReasoningAgent:

    def __init__(self):

        self.name = "Clinical Reasoning Agent"

        self.model_name = (
            "Rule-Based Multi-Agent Clinical Reasoning"
        )

    def predict(self, agent_results):

        try:

            # ==================================================
            # RETRIEVE AGENT RESULTS
            # ==================================================

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

            tumor = agent_results.get(
                "tumor_classification",
                {}
            )

            segmentation = agent_results.get(
                "liver_segmentation",
                {}
            )

            # ==================================================
            # PREDICTIONS
            # ==================================================

            fatty_prediction = fatty.get(
                "prediction"
            )

            fibrosis_prediction = fibrosis.get(
                "prediction"
            )

            cirrhosis_prediction = cirrhosis.get(
                "prediction"
            )

            tumor_prediction = tumor.get(
                "prediction"
            )

            # ==================================================
            # FINDINGS
            # ==================================================

            findings = []

            if fatty.get("status") == "completed":

                findings.append({
                    "domain": "fatty_liver",
                    "prediction":
                        fatty_prediction,
                    "confidence":
                        fatty.get("probability")
                })

            if fibrosis.get("status") == "completed":

                findings.append({
                    "domain": "fibrosis",
                    "prediction":
                        fibrosis_prediction,
                    "confidence":
                        fibrosis.get("probability")
                })

            if cirrhosis.get("status") == "completed":

                findings.append({
                    "domain": "cirrhosis",
                    "prediction":
                        cirrhosis_prediction,
                    "confidence":
                        cirrhosis.get("probability")
                })

            if tumor.get("status") == "completed":

                findings.append({
                    "domain": "tumor",
                    "prediction":
                        tumor_prediction,
                    "confidence":
                        tumor.get("probability")
                })

            if segmentation.get(
                "status"
            ) == "completed":

                findings.append({
                    "domain": "segmentation",
                    "prediction":
                        segmentation.get(
                            "prediction"
                        ),
                    "liver_percentage":
                        segmentation.get(
                            "liver_percentage"
                        )
                })

            # ==================================================
            # TUMOR FLAG
            # ==================================================

            tumor_detected = False

            if tumor_prediction:

                tumor_detected = (
                    str(
                        tumor_prediction
                    ).lower()
                    not in [
                        "healthy",
                        "normal",
                        "none"
                    ]
                )

            # ==================================================
            # RISK SCORE
            # ==================================================

            risk_score = 0

            # Cirrhosis
            if str(
                cirrhosis_prediction
            ) in ["1", "2", "3"]:

                risk_score += 3

            # Fibrosis
            if str(
                fibrosis_prediction
            ) in ["1", "2", "3"]:

                risk_score += 2

            # Fatty liver
            if str(
                fatty_prediction
            ) in ["1", "2"]:

                risk_score += 1

            # Tumor
            if tumor_detected:

                risk_score += 4

            # ==================================================
            # OVERALL RISK
            # ==================================================

            if risk_score >= 6:

                overall_risk = "High"

            elif risk_score >= 3:

                overall_risk = "Moderate"

            elif risk_score >= 1:

                overall_risk = "Low"

            else:

                overall_risk = "No major abnormality detected"

            # ==================================================
            # UNIFIED ASSESSMENT
            # ==================================================

            assessment = {

                "overall_risk":
                    overall_risk,

                "risk_score":
                    risk_score,

                "fatty_liver":
                    fatty_prediction,

                "fibrosis":
                    fibrosis_prediction,

                "cirrhosis":
                    cirrhosis_prediction,

                "tumor":
                    tumor_prediction,

                "tumor_detected":
                    tumor_detected,

                "segmentation":
                    {
                        "available":
                            segmentation.get(
                                "segmentation_available",
                                False
                            ),
                        "liver_percentage":
                            segmentation.get(
                                "liver_percentage"
                            )
                    },

                "findings":
                    findings
            }

            # ==================================================
            # FINAL RESULT
            # ==================================================

            return {

                "agent": self.name,

                "model": self.model_name,

                "status": "completed",

                "unified_assessment":
                    assessment,

                "findings":
                    findings,

                "overall_risk":
                    overall_risk
            }

        except Exception as e:

            return {

                "agent": self.name,

                "model": self.model_name,

                "status": "error",

                "error": str(e)
            }
