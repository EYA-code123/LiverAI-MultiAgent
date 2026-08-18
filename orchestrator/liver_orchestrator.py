

# ==========================================================
# LiverAI - Multi-Agent Orchestrator
# ==========================================================

from agents.fatty_liver_agent import FattyLiverAgent
from agents.fibrosis_agent import FibrosisAgent
from agents.cirrhosis_agent import CirrhosisAgent
from agents.clinical_reasoning_agent import ClinicalReasoningAgent


class LiverAIOrchestrator:

    def __init__(
        self,
        fatty_model_path,
        fibrosis_model_path,
        cirrhosis_model_path
    ):

        self.name = "LiverAIOrchestrator"

        print("=" * 70)
        print("INITIALIZING LIVERAI ORCHESTRATOR")
        print("=" * 70)

        # --------------------------------------------------
        # Specialized agents
        # --------------------------------------------------

        print("\nLoading models...")

        self.fatty_agent = FattyLiverAgent(
            fatty_model_path
        )
        print("✓ Fatty Liver Agent initialized")

        self.fibrosis_agent = FibrosisAgent(
            fibrosis_model_path
        )
        print("✓ Fibrosis Agent initialized")

       import joblib

cirrhosis_model_package = joblib.load(
    cirrhosis_model_path
)

self.cirrhosis_agent = CirrhosisAgent(
    cirrhosis_model_package
)
        print("✓ Cirrhosis Agent initialized")

        # --------------------------------------------------
        # Clinical reasoning
        # --------------------------------------------------

        self.clinical_reasoning_agent = ClinicalReasoningAgent()

        print("✓ Clinical Reasoning Agent initialized")

        print("\n✓ LiverAI Orchestrator ready")

    # ======================================================
    # PREDICTION
    # ======================================================

    def predict(self, patient_data):

        print("\n")
        print("=" * 70)
        print("LIVERAI MULTI-AGENT PREDICTION")
        print("=" * 70)

        # --------------------------------------------------
        # 1. Fatty Liver Agent
        # --------------------------------------------------

        print("\n[1/4] Running Fatty Liver Agent...")

        fatty_result = self.fatty_agent.predict(
            patient_data
        )

        print("✓ Fatty Liver completed")

        # --------------------------------------------------
        # 2. Fibrosis Agent
        # --------------------------------------------------

        print("\n[2/4] Running Fibrosis Agent...")

        fibrosis_result = self.fibrosis_agent.predict(
            patient_data
        )

        print("✓ Fibrosis completed")

        # --------------------------------------------------
        # 3. Cirrhosis Agent
        # --------------------------------------------------

        print("\n[3/4] Running Cirrhosis Agent...")

        cirrhosis_result = self.cirrhosis_agent.predict(
            patient_data
        )

        print("✓ Cirrhosis completed")

        # --------------------------------------------------
        # Shared patient context
        # --------------------------------------------------

        agent_results = {
            "fatty_liver": fatty_result,
            "fibrosis": fibrosis_result,
            "cirrhosis": cirrhosis_result
        }

        # --------------------------------------------------
        # 4. Clinical Reasoning Agent
        # --------------------------------------------------

        print("\n[4/4] Running Clinical Reasoning Agent...")

        clinical_result = (
            self.clinical_reasoning_agent.predict(
                agent_results
            )
        )

        print("✓ Clinical Reasoning completed")

        # --------------------------------------------------
        # Final result
        # --------------------------------------------------

        return {
            "fatty_liver": fatty_result,
            "fibrosis": fibrosis_result,
            "cirrhosis": cirrhosis_result,
            "clinical_reasoning": clinical_result
        }
