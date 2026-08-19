import os
import joblib

from agents.fatty_liver_agent import FattyLiverAgent
from agents.fibrosis_agent import FibrosisAgent
from agents.cirrhosis_agent import CirrhosisAgent


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

        print("\nLoading models...")

        # ==================================================
        # 1. FATTY LIVER MODEL
        # ==================================================

        if not os.path.exists(fatty_model_path):
            raise FileNotFoundError(
                f"Fatty Liver model not found: {fatty_model_path}"
            )

        fatty_model = joblib.load(
            fatty_model_path
        )

        # If a package/dictionary was saved
        if isinstance(fatty_model, dict):
            fatty_model = fatty_model.get(
                "model",
                fatty_model
            )

        self.fatty_agent = FattyLiverAgent(
            fatty_model
        )

        print("✓ Fatty Liver Agent initialized")

        # ==================================================
        # 2. FIBROSIS MODEL
        # ==================================================

        if not os.path.exists(fibrosis_model_path):
            raise FileNotFoundError(
                f"Fibrosis model not found: {fibrosis_model_path}"
            )

        fibrosis_package = joblib.load(
            fibrosis_model_path
        )

        # Your fibrosis file is a package containing:
        # model + features + encoders

        if isinstance(fibrosis_package, dict):

            fibrosis_model = fibrosis_package["model"]

        else:

            fibrosis_model = fibrosis_package

        self.fibrosis_agent = FibrosisAgent(
            fibrosis_model
        )

        print("✓ Fibrosis Agent initialized")

        # ==================================================
        # 3. CIRRHOSIS MODEL
        # ==================================================

        if not os.path.exists(cirrhosis_model_path):
            raise FileNotFoundError(
                f"Cirrhosis model not found: {cirrhosis_model_path}"
            )

        cirrhosis_package = joblib.load(
            cirrhosis_model_path
        )

        self.cirrhosis_agent = CirrhosisAgent(
            cirrhosis_package
        )

        print("✓ Cirrhosis Agent initialized")

        # ==================================================
        # 4. CLINICAL REASONING
        # ==================================================

        from agents.clinical_reasoning_agent import (
            ClinicalReasoningAgent
        )

        self.clinical_reasoning_agent = (
            ClinicalReasoningAgent()
        )

        print("✓ Clinical Reasoning Agent initialized")

        print("\n✓ LiverAI Orchestrator ready")
