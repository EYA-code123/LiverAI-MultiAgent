class ClinicalReasoningAgent:

    def __init__(self):

        self.name = (
            "ClinicalReasoningAgent"
        )

        self.model_name = (
            "Rule-Based Clinical Reasoning"
        )

    # ======================================================
    # PREDICT
    # ======================================================

    def predict(
        self,
        agent_results
    ):

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
        # EXTRACT PREDICTIONS
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

            findings.append(
                f"Fatty liver prediction: "
                f"{fatty_prediction}"
            )

        if fibrosis.get("status") == "completed":

            findings.append(
                f"Fibrosis prediction: "
                f"{fibrosis_prediction}"
            )

        if cirrhosis.get("status") == "completed":

            findings.append(
                f"Cirrhosis prediction: "
                f"{cirrhosis_prediction}"
            )

        if tumor.get("status") in [
            "success",
            "completed"
        ]:

            findings.append(
                f"Tumor classification: "
                f"{tumor_prediction}"
            )

        if segmentation.get(
            "segmentation_available",
            False
        ):

            findings.append(
                "Liver segmentation "
                "successfully completed."
            )

            findings.append(
                "Estimated liver volume "
                f"percentage: "
                f"{segmentation.get('liver_percentage', 0):.2f}%"
            )

        # ==================================================
        # RISK LOGIC
        # ==================================================

        overall_risk = "Low"

        # ------------------------------------------
        # CIRRHOSIS
        # ------------------------------------------

        if str(
            cirrhosis_prediction
        ) in [
            "1",
            "2"
        ]:

            overall_risk = "Elevated"

        # ------------------------------------------
        # FIBROSIS
        # ------------------------------------------

        elif str(
            fibrosis_prediction
        ) == "1":

            overall_risk = "Moderate"

        # ------------------------------------------
        # FATTY LIVER
        # ------------------------------------------

        elif str(
            fatty_prediction
        ) in [
            "1",
            "2"
        ]:

            overall_risk = (
                "Possible fatty liver"
            )

        # ==================================================
        # TUMOR
        # ==================================================

        tumor_flag = False

        if tumor_prediction is not None:

            if str(
                tumor_prediction
            ).lower() not in [
                "healthy",
                "none",
                "normal"
            ]:

                tumor_flag = True

        if tumor_flag:

            overall_risk = "Requires tumor assessment"

        # ==================================================
        # RESULT
        # ==================================================

        return {

            "agent":
                self.name,

            "model":
                self.model_name,

            "overall_risk":
                overall_risk,

            "findings":
                findings,

            "tumor_detected":
                tumor_flag,

            "fatty_liver_prediction":
                fatty_prediction,

            "fibrosis_prediction":
                fibrosis_prediction,

            "cirrhosis_prediction":
                cirrhosis_prediction,

            "tumor_prediction":
                tumor_prediction,

            "segmentation_available":
                segmentation.get(
                    "segmentation_available",
                    False
                ),

            "status":
                "completed"
        }
