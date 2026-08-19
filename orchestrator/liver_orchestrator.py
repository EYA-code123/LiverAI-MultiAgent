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

        # ==================================================
        # CHECK MODEL PATHS
        # ==================================================

        paths = {
            "Fatty Liver": fatty_model_path,
            "Fibrosis": fibrosis_model_path,
            "Cirrhosis": cirrhosis_model_path
        }

        print("\nChecking model paths...")

        for name, path in paths.items():

            print(f"\n{name}:")
            print(path)

            if not os.path.exists(path):

                raise FileNotFoundError(
                    f"{name} model not found:\n{path}"
                )

            print("✓ Exists")

        # ==================================================
        # LOAD FATTY LIVER MODEL
        # ==================================================

        print("\nLoading Fatty Liver model...")

        fatty_model = joblib.load(
            fatty_model_path
        )

        print(
            "Loaded:",
            type(fatty_model)
        )

        if not hasattr(
            fatty_model,
            "predict"
        ):

            raise TypeError(
                "Fatty Liver model does not have predict()."
            )

        self.fatty_agent = FattyLiverAgent(
            fatty_model
        )

        print(
            "✓ Fatty Liver Agent initialized"
        )

        # ==================================================
        # LOAD FIBROSIS MODEL
        # ==================================================

        print("\nLoading Fibrosis model...")

        fibrosis_package = joblib.load(
            fibrosis_model_path
        )

        print(
            "Loaded:",
            type(fibrosis_package)
        )

        if not isinstance(
            fibrosis_package,
            dict
        ):

            raise TypeError(
                "Fibrosis model must be a dictionary package."
            )

        if "model" not in fibrosis_package:

            raise KeyError(
                "Fibrosis package does not contain 'model'."
            )

        fibrosis_model = (
            fibrosis_package["model"]
        )

        if not hasattr(
            fibrosis_model,
            "predict"
        ):

            raise TypeError(
                "Fibrosis internal model does not have predict()."
            )

        self.fibrosis_agent = FibrosisAgent(
            fibrosis_package
        )

        print(
            "✓ Fibrosis Agent initialized"
        )

        # ==================================================
        # LOAD CIRRHOSIS MODEL
        # ==================================================

        print("\nLoading Cirrhosis model...")

        cirrhosis_package = joblib.load(
            cirrhosis_model_path
        )

        print(
            "Loaded:",
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

        cirrhosis_model = (
            cirrhosis_package["model"]
        )

        if not hasattr(
            cirrhosis_model,
            "predict"
        ):

            raise TypeError(
                "Cirrhosis internal model does not have predict()."
            )

        self.cirrhosis_agent = CirrhosisAgent(
            cirrhosis_package
        )

        print(
            "✓ Cirrhosis Agent initialized"
        )

        # ==================================================
        # CLINICAL REASONING AGENT
        # ==================================================

        self.clinical_reasoning_agent = (
            ClinicalReasoningAgent()
        )

        print(
            "✓ Clinical Reasoning Agent initialized"
        )

        # ==================================================
        # READY
        # ==================================================

        print("\n✓ LiverAI Orchestrator ready")

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
        # SHARED AGENT RESULTS
        # ==================================================

        agent_results = {

            "fatty_liver":
                fatty_result,

            "fibrosis":
                fibrosis_result,

            "cirrhosis":
                cirrhosis_result

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

        result = {

            "fatty_liver":
                fatty_result,

            "fibrosis":
                fibrosis_result,

            "cirrhosis":
                cirrhosis_result,

            "clinical_reasoning":
                clinical_result

        }

        print("\n")
        print("=" * 70)
        print("LIVERAI MULTI-AGENT RESULT")
        print("=" * 70)

        return result
