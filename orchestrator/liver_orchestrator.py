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
        # 1. LOAD FATTY LIVER MODEL
        # ==========================================================

        print("\nLoading Fatty Liver model...")

        if not os.path.exists(fatty_model_path):
            raise FileNotFoundError(
                f"Fatty Liver model not found:\n{fatty_model_path}"
            )

        fatty_model = joblib.load(
            fatty_model_path
        )

        # Si le fichier contient directement le modèle
        self.fatty_agent = FattyLiverAgent(
            fatty_model
        )

        print("✓ Fatty Liver Agent initialized")
        print(
            "  Model type:",
            type(fatty_model)
        )

        # ==========================================================
        # 2. LOAD FIBROSIS MODEL
        # ==========================================================

        print("\nLoading Fibrosis model...")

        if not os.path.exists(fibrosis_model_path):
            raise FileNotFoundError(
                f"Fibrosis model not found:\n{fibrosis_model_path}"
            )

        fibrosis_package = joblib.load(
            fibrosis_model_path
        )

        # Ton fichier Fibrosis contient un package
        if isinstance(
            fibrosis_package,
            dict
        ):

            fibrosis_model = fibrosis_package["model"]

        else:

            fibrosis_model = fibrosis_package

        self.fibrosis_agent = FibrosisAgent(
            fibrosis_model
        )

        print("✓ Fibrosis Agent initialized")
        print(
            "  Model type:",
            type(fibrosis_model)
        )

        # ==========================================================
        # 3. LOAD CIRRHOSIS MODEL
        # ==========================================================

        print("\nLoading Cirrhosis model...")

        if not os.path.exists(cirrhosis_model_path):
            raise FileNotFoundError(
                f"Cirrhosis model not found:\n{cirrhosis_model_path}"
            )

        cirrhosis_package = joblib.load(
            cirrhosis_model_path
        )

        self.cirrhosis_agent = CirrhosisAgent(
            cirrhosis_package
        )

        print("✓ Cirrhosis Agent initialized")

        # ==========================================================
        # 4. CLINICAL REASONING
        # ==========================================================

        self.clinical_reasoning_agent = (
            ClinicalReasoningAgent()
        )

        print("✓ Clinical Reasoning Agent initialized")

        print("\n✓ LiverAI Orchestrator ready")
