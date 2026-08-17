%%writefile /content/LiverAI-MultiAgent/orchestrator/liver_orchestrator.py

# ==========================================================
# LiverAI Multi-Agent Orchestrator
# ==========================================================

import os
import sys
import joblib


PROJECT_PATH = "/content/LiverAI-MultiAgent"

if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)


# ==========================================================
# IMPORT AGENTS
# ==========================================================

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

        # ==================================================
        # CHECK MODELS
        # ==================================================

        paths = {
            "Fatty Liver": fatty_model_path,
            "Fibrosis": fibrosis_model_path,
            "Cirrhosis": cirrhosis_model_path
        }

        for name, path in paths.items():

            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{name} model not found:\n{path}"
                )

            print(f"✓ {name} model found")

        # ==================================================
        # LOAD MODELS
        # ==================================================

        print("\nLoading models...")

        fatty_model = joblib.load(
            fatty_model_path
        )

        fibrosis_model = joblib.load(
            fibrosis_model_path
        )

        # IMPORTANT:
        # Cirrhosis model is a package/dictionary
        cirrhosis_package = joblib.load(
            cirrhosis_model_path
        )

        print("✓ Fatty Liver model loaded")
        print("✓ Fibrosis model loaded")
        print("✓ Cirrhosis package loaded")

        # ==================================================
        # CREATE AGENTS
        # ==================================================

        self.fatty_agent = FattyLiverAgent(
            fatty_model
        )

        self.fibrosis_agent = FibrosisAgent(
            fibrosis_model
        )

        self.cirrhosis_agent = CirrhosisAgent(
            cirrhosis_package
        )

        print("\n✓ Fatty Liver Agent initialized")
        print("✓ Fibrosis Agent initialized")
        print("✓ Cirrhosis Agent initialized")

        print("\n✓ LiverAI Orchestrator ready")

    # ======================================================
    # PREDICTION
    # ======================================================

    def predict(
        self,
        fatty_data,
        fibrosis_data,
        cirrhosis_data
    ):

        print("\n")
        print("=" * 70)
        print("LIVERAI MULTI-AGENT PREDICTION")
        print("=" * 70)

        results = {}

        # ==================================================
        # 1. FATTY LIVER
        # ==================================================

        print("\n[1/3] Running Fatty Liver Agent...")

        try:

            fatty_result = self.fatty_agent.predict(
                fatty_data
            )

            results["fatty_liver"] = fatty_result

            print("✓ Fatty Liver completed")

        except Exception as e:

            results["fatty_liver"] = {
                "agent": "FattyLiverAgent",
                "status": "error",
                "error": str(e)
            }

            print("✗ Fatty Liver error:", e)

        # ==================================================
        # 2. FIBROSIS
        # ==================================================

        print("\n[2/3] Running Fibrosis Agent...")

        try:

            fibrosis_result = self.fibrosis_agent.predict(
                fibrosis_data
            )

            results["fibrosis"] = fibrosis_result

            print("✓ Fibrosis completed")

        except Exception as e:

            results["fibrosis"] = {
                "agent": "FibrosisAgent",
                "status": "error",
                "error": str(e)
            }

            print("✗ Fibrosis error:", e)

        # ==================================================
        # 3. CIRRHOSIS
        # ==================================================

        print("\n[3/3] Running Cirrhosis Agent...")

        try:

            cirrhosis_result = self.cirrhosis_agent.predict(
                cirrhosis_data
            )

            results["cirrhosis"] = cirrhosis_result

            print("✓ Cirrhosis completed")

        except Exception as e:

            results["cirrhosis"] = {
                "agent": "CirrhosisAgent",
                "status": "error",
                "error": str(e)
            }

            print("✗ Cirrhosis error:", e)

        # ==================================================
        # BUILD SUMMARY
        # ==================================================

        return self._build_summary(results)

    # ======================================================
    # SUMMARY
    # ======================================================

    def _build_summary(self, results):

        summary = {
            "fatty_liver": results.get(
                "fatty_liver"
            ),

            "fibrosis": results.get(
                "fibrosis"
            ),

            "cirrhosis": results.get(
                "cirrhosis"
            )
        }

        successful_agents = 0

        for result in summary.values():

            if (
                result is not None
                and result.get("status") == "completed"
            ):

                successful_agents += 1

        summary["agents_completed"] = successful_agents

        summary["total_agents"] = 3

        return summary


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

def print_results(results):

    print("\n")
    print("=" * 70)
    print("LIVERAI MULTI-AGENT RESULTS")
    print("=" * 70)

    for name in [
        "fatty_liver",
        "fibrosis",
        "cirrhosis"
    ]:

        result = results.get(name)

        print(f"\n{name.upper()}")

        if result:

            print(
                "Prediction :",
                result.get("prediction")
            )

            print(
                "Probability:",
                result.get("probability")
            )

            print(
                "Status     :",
                result.get("status")
            )

    print("\n" + "-" * 70)

    print(
        f"Agents completed: "
        f"{results.get('agents_completed')}/"
        f"{results.get('total_agents')}"
    )

    print("=" * 70)
