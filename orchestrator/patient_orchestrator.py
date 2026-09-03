from typing import Dict, Any


class PatientOrchestrator:

    def __init__(
        self,
        cirrhosis_agent=None,
        fatty_liver_agent=None,
        fibrosis_agent=None,
        clinical_agent=None,
        tumor_agent=None,
        segmentation_agent=None,
    ):
        self.agents = {
            "cirrhosis": cirrhosis_agent,
            "fatty_liver": fatty_liver_agent,
            "fibrosis": fibrosis_agent,
            "clinical_reasoning": clinical_agent,
            "tumor": tumor_agent,
            "segmentation": segmentation_agent,
        }

    def run(
        self,
        clinical_data=None,
        fibrosis_data=None,
        image=None,
        volume_3d=None,
    ) -> Dict[str, Any]:

        results = {}

        # ============================================================
        # 1. CIRRHOSIS
        # ============================================================
        if clinical_data is not None:
            agent = self.agents["cirrhosis"]

            if agent is not None:
                try:
                    results["cirrhosis"] = agent.predict(clinical_data)
                except Exception as e:
                    results["cirrhosis"] = {
                        "status": "error",
                        "error": str(e),
                    }

        # ============================================================
        # 2. FATTY LIVER
        # ============================================================
        if clinical_data is not None:
            agent = self.agents["fatty_liver"]

            if agent is not None:
                try:
                    results["fatty_liver"] = agent.predict(clinical_data)
                except Exception as e:
                    results["fatty_liver"] = {
                        "status": "error",
                        "error": str(e),
                    }

        # ============================================================
        # 3. FIBROSIS
        # ============================================================
        if fibrosis_data is not None:
            agent = self.agents["fibrosis"]

            if agent is not None:
                try:
                    results["fibrosis"] = agent.predict(fibrosis_data)
                except Exception as e:
                    results["fibrosis"] = {
                        "status": "error",
                        "error": str(e),
                    }

        # ============================================================
        # 4. TUMOR
        # ============================================================
        if image is not None:
            agent = self.agents["tumor"]

            if agent is not None:
                try:
                    results["tumor"] = agent.predict(image)
                except Exception as e:
                    results["tumor"] = {
                        "status": "error",
                        "error": str(e),
                    }

        # ============================================================
        # 5. SEGMENTATION
        # ============================================================
        if volume_3d is not None:
            agent = self.agents["segmentation"]

            if agent is not None:
                try:
                    results["segmentation"] = agent.predict(volume_3d)
                except Exception as e:
                    results["segmentation"] = {
                        "status": "error",
                        "error": str(e),
                    }
            else:
                results["segmentation"] = {
                    "status": "unavailable",
                    "reason": "Segmentation model not configured",
                }

        # ============================================================
        # 6. CLINICAL REASONING
        # ============================================================
        agent = self.agents["clinical_reasoning"]

        if agent is not None and clinical_data is not None:
            try:
                results["clinical_reasoning"] = agent.predict(
                    clinical_data
                )
            except Exception as e:
                results["clinical_reasoning"] = {
                    "status": "error",
                    "error": str(e),
                }

        return self._build_patient_summary(results)

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    def _build_patient_summary(self, results):

        available = []
        errors = []
        unavailable = []

        for name, result in results.items():

            if not isinstance(result, dict):
                available.append(name)
                continue

            status = result.get("status")

            if status == "error":
                errors.append(name)

            elif status == "unavailable":
                unavailable.append(name)

            else:
                available.append(name)

        return {
            "patient_assessment": results,
            "coordination": {
                "available_agents": available,
                "failed_agents": errors,
                "unavailable_agents": unavailable,
            },
        } 
