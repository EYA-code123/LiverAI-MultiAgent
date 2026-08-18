

import os

from agents.fatty_liver_agent import FattyLiverAgent
from agents.fibrosis_agent import FibrosisAgent
from agents.cirrhosis_agent import CirrhosisAgent
from agents.clinical_reasoning_agent import ClinicalReasoningAgent


class LiverAIOrchestrator:

    def __init__(
        self,
        fatty_model_path,
        fibrosis_model_path,
        cirrhosis_model_path,
        fibrosis_encoder_path=None
    ):

        print("=" * 70)
        print("INITIALIZING LIVERAI ORCHESTRATOR")
        print("=" * 70)

        print("✓ Fatty Liver model found")
        print("✓ Fibrosis model found")
        print("✓ Cirrhosis model found")

        print("\nLoading models...\n")

        self.fatty_agent = FattyLiverAgent(
            fatty_model_path
        )

        self.fibrosis_agent = FibrosisAgent(
            fibrosis_model_path,
            fibrosis_encoder_path
        )

        self.cirrhosis_agent = CirrhosisAgent(
            cirrhosis_model_path
        )

        self.reasoning_agent = ClinicalReasoningAgent()

        print("✓ Fatty Liver Agent initialized")
        print("✓ Fibrosis Agent initialized")
        print("✓ Cirrhosis Agent initialized")
        print("✓ Clinical Reasoning Agent initialized")

        print("\n✓ LiverAI Orchestrator ready")

    def predict(self, patient):

        print("\n")
        print("=" * 70)
        print("LIVERAI MULTI-AGENT PREDICTION")
        print("=" * 70)

        results = {}

        # --------------------------------------------------
        # 1. FATty LIVER
        # --------------------------------------------------

        print("\n[1/4] Running Fatty Liver Agent...")

        try:
            results["fatty_liver"] = self.fatty_agent.predict(patient)
            print("✓ Fatty Liver completed")

        except Exception as e:
            results["fatty_liver"] = {
                "agent": "FattyLiverAgent",
                "status": "error",
                "error": str(e)
            }
            print("✗ Fatty Liver failed:", e)

        # --------------------------------------------------
        # 2. FIBROSIS
        # --------------------------------------------------

        print("\n[2/4] Running Fibrosis Agent...")

        try:
            results["fibrosis"] = self.fibrosis_agent.predict(patient)
            print("✓ Fibrosis completed")

        except Exception as e:
            results["fibrosis"] = {
                "agent": "FibrosisAgent",
                "status": "error",
                "error": str(e)
            }
            print("✗ Fibrosis failed:", e)

        # --------------------------------------------------
        # 3. CIRRHOSIS
        # --------------------------------------------------

        print("\n[3/4] Running Cirrhosis Agent...")

        try:
            results["cirrhosis"] = self.cirrhosis_agent.predict(patient)
            print("✓ Cirrhosis completed")

        except Exception as e:
            results["cirrhosis"] = {
                "agent": "CirrhosisAgent",
                "status": "error",
                "error": str(e)
            }
            print("✗ Cirrhosis failed:", e)

        # --------------------------------------------------
        # 4. CLINICAL REASONING
        # --------------------------------------------------

        print("\n[4/4] Running Clinical Reasoning Agent...")

        try:
            clinical = self.reasoning_agent.predict(results)

            results["clinical_reasoning"] = clinical

            print("✓ Clinical Reasoning completed")

        except Exception as e:

            results["clinical_reasoning"] = {
                "agent": "ClinicalReasoningAgent",
                "status": "error",
                "error": str(e)
            }

            print("✗ Clinical Reasoning failed:", e)

        # --------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------

        results["status"] = "completed"

        results["agents_completed"] = sum(
            1
            for key in [
                "fatty_liver",
                "fibrosis",
                "cirrhosis",
                "clinical_reasoning"
            ]
            if results.get(key, {}).get("status") == "completed"
        )

        results["total_agents"] = 4

        return results
