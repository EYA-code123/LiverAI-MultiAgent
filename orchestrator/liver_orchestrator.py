# ==========================================================
# LIVER AI ORCHESTRATOR
# ==========================================================

import os
import joblib


# ==========================================================
# AGENTS
# ==========================================================

from agents.fatty_liver_agent import FattyLiverAgent
from agents.fibrosis_agent import FibrosisAgent
from agents.cirrhosis_agent import CirrhosisAgent
from agents.clinical_reasoning_agent import ClinicalReasoningAgent


# ==========================================================
# ORCHESTRATOR
# ==========================================================

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

        # ==================================================
        # CHECK PATHS
        # ==================================================

        print("\nLoading models...")

        if not os.path.exists(fatty_model_path):
            raise FileNotFoundError(
                f"Fatty Liver model not found:\n{fatty_model_path}"
            )

        if not os.path.exists(fibrosis_model_path):
            raise FileNotFoundError(
                f"Fibrosis model not found:\n{fibrosis_model_path}"
            )

        if not os.path.exists(cirrhosis_model_path):
            raise FileNotFoundError(
                f"Cirrhosis model not found:\n{cirrhosis_model_path}"
            )

        # ==================================================
        # 1. LOAD FATTY LIVER MODEL
        # ==================================================

        print("\n[1/4] Loading Fatty Liver model...")

        fatty_model = joblib.load(
            fatty_model_path
        )

        print(
            "Fatty model loaded:",
            type(fatty_model)
        )

        if not hasattr(fatty_model, "predict"):

            raise TypeError(
                "Fatty Liver model does not have a predict() method.\n"
                f"Loaded type: {type(fatty_model)}"
            )

        self.fatty_agent = FattyLiverAgent(
            fatty_model
        )

        print(
            "✓ Fatty Liver Agent initialized"
        )

        # ==================================================
        # 2. LOAD FIBROSIS MODEL
        # ==================================================

        print("\n[2/4] Loading Fibrosis model...")

        fibrosis_package = joblib.load(
            fibrosis_model_path
        )

        print(
            "Fibrosis package type:",
            type(fibrosis_package)
        )

        # --------------------------------------------------
        # Fibrosis model was saved as a dictionary package
        # --------------------------------------------------

        if isinstance(
            fibrosis_package,
            dict
        ):

            if "model" not in fibrosis_package:

                raise KeyError(
                    "Fibrosis model package does not contain "
                    "'model'."
                )

            fibrosis_model = fibrosis_package["model"]

        else:

            # In case the model was saved directly
            fibrosis_model = fibrosis_package

        print(
            "Fibrosis model loaded:",
            type(fibrosis_model)
        )

        if not hasattr(
            fibrosis_model,
            "predict"
        ):

            raise TypeError(
                "Fibrosis model does not have a predict() method.\n"
                f"Loaded type: {type(fibrosis_model)}"
            )

        self.fibrosis_agent = FibrosisAgent(
            fibrosis_model
        )

        print(
            "✓ Fibrosis Agent initialized"
        )

        # ==================================================
        # 3. LOAD CIRRHOSIS MODEL
        # ==================================================

        print("\n[3/4] Loading Cirrhosis model...")

        cirrhosis_package = joblib.load(
            cirrhosis_model_path
        )

        print(
            "Cirrhosis package type:",
            type(cirrhosis_package)
        )

        if not isinstance(
            cirrhosis_package,
            dict
        ):

            raise TypeError(
                "Cirrhosis model must be a dictionary package."
            )

        if "model" not in cirrhosis_package:

            raise KeyError(
                "Cirrhosis package does not contain 'model'."
            )

        self.cirrhosis_agent = CirrhosisAgent(
            cirrhosis_package
        )

        print(
            "✓ Cirrhosis Agent initialized"
        )

        # ==================================================
        # 4. CLINICAL REASONING AGENT
        # ==================================================

        print("\n[4/4] Initializing Clinical Reasoning Agent...")

        self.clinical_reasoning_agent = (
            ClinicalReasoningAgent()
        )

        print(
            "✓ Clinical Reasoning Agent initialized"
        )

        # ==================================================
        # FINAL CHECK
        # ==================================================

        print("\n" + "=" * 70)
        print("FINAL MODEL TYPE CHECK")
        print("=" * 70)

        print(
            "\nFatty model:",
            type(self.fatty_agent.model)
        )

        print(
            "Fatty has predict:",
            hasattr(
                self.fatty_agent.model,
                "predict"
            )
        )

        print(
            "\nFibrosis model:",
            type(self.fibrosis_agent.model)
        )

        print(
            "Fibrosis has predict:",
            hasattr(
                self.fibrosis_agent.model,
                "predict"
            )
        )

        print(
            "\nCirrhosis model:",
            type(self.cirrhosis_agent.model)
        )

        print(
            "Cirrhosis has predict:",
            hasattr(
                self.cirrhosis_agent.model,
                "predict"
            )
        )

        print("\n" + "=" * 70)
        print("✓ LIVERAI ORCHESTRATOR READY")
        print("=" * 70)

    # ======================================================
    # PREDICT
    # ======================================================

    def predict(
        self,
        patient_data
    ):

        print("\n")
        print("=" * 70)
        print("LIVERAI MULTI-AGENT PREDICTION")
        print("=" * 70)

        # ==================================================
        # 1. FATTY LIVER
        # ==================================================

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

        # ==================================================
        # 2. FIBROSIS
        # ==================================================

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

        # ==================================================
        # 3. CIRRHOSIS
        # ==================================================

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

        # ==================================================
        # SHARED RESULTS
        # ==================================================

        agent_results = {

            "fatty_liver": fatty_result,

            "fibrosis": fibrosis_result,

            "cirrhosis": cirrhosis_result

        }

        # ==================================================
        # 4. CLINICAL REASONING
        # ==================================================

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

        # ==================================================
        # FINAL RESULT
        # ==================================================

        final_result = {

            "fatty_liver": fatty_result,

            "fibrosis": fibrosis_result,

            "cirrhosis": cirrhosis_result,

            "clinical_reasoning": clinical_result

        }

        print("\n" + "=" * 70)
        print("LIVERAI MULTI-AGENT PREDICTION COMPLETED")
        print("=" * 70)

        return final_result
