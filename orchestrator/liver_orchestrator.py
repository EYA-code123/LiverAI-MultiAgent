import os
import joblib

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

        print("=" * 70)
        print("INITIALIZING LIVERAI ORCHESTRATOR")
        print("=" * 70)

        # ==========================================================
        # CHECK PATHS
        # ==========================================================

        print("\nLoading models...")

        if not os.path.exists(
            fatty_model_path
        ):

            raise FileNotFoundError(
                f"Fatty Liver model not found:\n"
                f"{fatty_model_path}"
            )

        if not os.path.exists(
            fibrosis_model_path
        ):

            raise FileNotFoundError(
                f"Fibrosis model not found:\n"
                f"{fibrosis_model_path}"
            )

        if not os.path.exists(
            cirrhosis_model_path
        ):

            raise FileNotFoundError(
                f"Cirrhosis model not found:\n"
                f"{cirrhosis_model_path}"
            )

        # ==========================================================
        # LOAD FATTY LIVER MODEL
        # ==========================================================

        print("\nLoading Fatty Liver model...")

        fatty_model = joblib.load(
            fatty_model_path
        )

        print(
            "Loaded Fatty model:",
            type(fatty_model)
        )

        self.fatty_agent = FattyLiverAgent(
            fatty_model
        )

        print(
            "✓ Fatty Liver Agent initialized"
        )

        # ==========================================================
        # LOAD FIBROSIS MODEL
        # ==========================================================

        print("\nLoading Fibrosis model...")

        fibrosis_package = joblib.load(
            fibrosis_model_path
        )

        print(
            "Loaded Fibrosis package:",
            type(fibrosis_package)
        )

        if isinstance(
            fibrosis_package,
            dict
        ):

            if "model" not in fibrosis_package:

                raise KeyError(
                    "Fibrosis package does not contain "
                    "'model'."
                )

            fibrosis_model = (
                fibrosis_package["model"]
            )

        else:

            fibrosis_model = fibrosis_package

        print(
            "Extracted Fibrosis model:",
            type(fibrosis_model)
        )

        self.fibrosis_agent = FibrosisAgent(
            fibrosis_model
        )

        print(
            "✓ Fibrosis Agent initialized"
        )

        # ==========================================================
        # LOAD CIRRHOSIS MODEL
        # ==========================================================

        print("\nLoading Cirrhosis model...")

        cirrhosis_package = joblib.load(
            cirrhosis_model_path
        )

        if not isinstance(
            cirrhosis_package,
            dict
        ):

            raise TypeError(
                "Cirrhosis model must be "
                "a dictionary package."
            )

        self.cirrhosis_agent = CirrhosisAgent(
            cirrhosis_package
        )

        print(
            "✓ Cirrhosis Agent initialized"
        )

        # ==========================================================
        # CLINICAL REASONING AGENT
        # ==========================================================

        self.clinical_reasoning_agent = (
            ClinicalReasoningAgent()
        )

        print(
            "✓ Clinical Reasoning Agent initialized"
        )

        print(
            "\n✓ LiverAI Orchestrator ready"
        )

    # ==============================================================
    # PREDICT
    # ==============================================================

    def predict(
        self,
        patient_data
    ):

        print("\n")
        print("=" * 70)
        print("LIVERAI MULTI-AGENT PREDICTION")
        print("=" * 70)

        # ==========================================================
        # 1. FATTY LIVER
        # ==========================================================

        print(
            "\n[1/4] Running Fatty Liver Agent..."
        )

        fatty_result = (
            self.fatty_agent.predict(
                patient_data
            )
        )

        print(
            "✓ Fatty Liver completed"
        )

        # ==========================================================
        # 2. FIBROSIS
        # ==========================================================

        print(
            "\n[2/4] Running Fibrosis Agent..."
        )

        fibrosis_result = (
            self.fibrosis_agent.predict(
                patient_data
            )
        )

        print(
            "✓ Fibrosis completed"
        )

        # ==========================================================
        # 3. CIRRHOSIS
        # ==========================================================

        print(
            "\n[3/4] Running Cirrhosis Agent..."
        )

        cirrhosis_result = (
            self.cirrhosis_agent.predict(
                patient_data
            )
        )

        print(
            "✓ Cirrhosis completed"
        )

        # ==========================================================
        # SHARED RESULTS
        # ==========================================================

        agent_results = {

            "fatty_liver":
                fatty_result,

            "fibrosis":
                fibrosis_result,

            "cirrhosis":
                cirrhosis_result
        }

        # ==========================================================
        # 4. CLINICAL REASONING
        # ==========================================================

        print(
            "\n[4/4] Running Clinical Reasoning Agent..."
        )

        clinical_result = (
            self.clinical_reasoning_agent.predict(
                agent_results
            )
        )

        print(
            "✓ Clinical Reasoning completed"
        )

        # ==========================================================
        # FINAL RESULT
        # ==========================================================

        return {

            "fatty_liver":
                fatty_result,

            "fibrosis":
                fibrosis_result,

            "cirrhosis":
                cirrhosis_result,

            "clinical_reasoning":
                clinical_result
        }
