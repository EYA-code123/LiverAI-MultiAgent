# ==========================================================
# LiverAI Multi-Agent Orchestrator
# ==========================================================

import os
import sys
import joblib
import numpy as np


# ==========================================================
# PROJECT PATH
# ==========================================================

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

        # --------------------------------------------------
        # Check model files
        # --------------------------------------------------

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


        # --------------------------------------------------
        # Load models
        # --------------------------------------------------

        print("\nLoading models...")

        fatty_model = joblib.load(
            fatty_model_path
        )

        fibrosis_model = joblib.load(
            fibrosis_model_path
        )

        cirrhosis_model = joblib.load(
            cirrhosis_model_path
        )


        # --------------------------------------------------
        # Create agents
        # --------------------------------------------------

        self.fatty_agent = FattyLiverAgent(
            fatty_model
        )

        self.fibrosis_agent = FibrosisAgent(
            fibrosis_model
        )

        self.cirrhosis_agent = CirrhosisAgent(
            cirrhosis_model
        )


        print("\n✓ Fatty Liver Agent initialized")
        print("✓ Fibrosis Agent initialized")
        print("✓ Cirrhosis Agent initialized")

        print("\n✓ Orchestrator ready")


    # ======================================================
    # RUN ALL AGENTS
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
        # RETURN COMBINED RESULTS
        # ==================================================

        return self._build_summary(
            results
        )


    # ======================================================
    # BUILD SUMMARY
    # ======================================================

    def _build_summary(
        self,
        results
    ):

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


        # --------------------------------------------------
        # Count successful agents
        # --------------------------------------------------

        successful_agents = 0

        for result in summary.values():

            if (
                result is not None
                and result.get("status") == "completed"
            ):

                successful_agents += 1


        summary["agents_completed"] = (
            successful_agents
        )

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


    # ------------------------------------------------------
    # Fatty Liver
    # ------------------------------------------------------

    fatty = results.get(
        "fatty_liver"
    )

    print("\nFATTY LIVER AGENT")

    if fatty:

        print(
            "Prediction :",
            fatty.get("prediction")
        )

        print(
            "Probability:",
            fatty.get("probability")
        )

        print(
            "Status     :",
            fatty.get("status")
        )


    # ------------------------------------------------------
    # Fibrosis
    # ------------------------------------------------------

    fibrosis = results.get(
        "fibrosis"
    )

    print("\nFIBROSIS AGENT")

    if fibrosis:

        print(
            "Prediction :",
            fibrosis.get("prediction")
        )

        print(
            "Probability:",
            fibrosis.get("probability")
        )

        print(
            "Status     :",
            fibrosis.get("status")
        )


    # ------------------------------------------------------
    # Cirrhosis
    # ------------------------------------------------------

    cirrhosis = results.get(
        "cirrhosis"
    )

    print("\nCIRRHOSIS AGENT")

    if cirrhosis:

        print(
            "Prediction :",
            cirrhosis.get("prediction")
        )

        print(
            "Probability:",
            cirrhosis.get("probability")
        )

        print(
            "Status     :",
            cirrhosis.get("status")
        )


    # ------------------------------------------------------
    # Global status
    # ------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        f"Agents completed: "
        f"{results.get('agents_completed')}/"
        f"{results.get('total_agents')}"
    )

    print("=" * 70)
