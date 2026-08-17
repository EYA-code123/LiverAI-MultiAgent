%%writefile /content/LiverAI-MultiAgent/orchestrator/liver_orchestrator.py

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

    # ======================================================
    # INITIALIZATION
    # ======================================================

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
        # Check models
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

        # Cirrhosis model is a package/dictionary
        cirrhosis_package = joblib.load(
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
            cirrhosis_package
        )


        print("\n✓ Fatty Liver Agent initialized")
        print("✓ Fibrosis Agent initialized")
        print("✓ Cirrhosis Agent initialized")

        print("\n✓ LiverAI Orchestrator ready")


    # ======================================================
    # PATIENT DATA PREPARATION
    # ======================================================

    def prepare_fatty_data(self, patient):

        return {
            "mcv": patient["mcv"],
            "alkphos": patient["alkphos"],
            "sgpt": patient["sgpt"],
            "sgot": patient["sgot"],
            "gammagt": patient["gammagt"],
            "drinks": patient["drinks"]
        }


    def prepare_fibrosis_data(self, patient):

        return np.array([
            patient["age"],
            patient["male"],
            patient["weight"],
            patient["height"],
            patient["bmi"],
            patient["futime"],
            patient["days"],
            patient["test"],
            patient["value"]
        ])


    def prepare_cirrhosis_data(self, patient):

        return {
            "N_Days": patient["N_Days"],
            "Status": patient["Status"],
            "Drug": patient["Drug"],
            "Age": patient["Age"],
            "Sex": patient["Sex"],
            "Ascites": patient["Ascites"],
            "Hepatomegaly": patient["Hepatomegaly"],
            "Spiders": patient["Spiders"],
            "Edema": patient["Edema"],
            "Bilirubin": patient["Bilirubin"],
            "Cholesterol": patient["Cholesterol"],
            "Albumin": patient["Albumin"],
            "Copper": patient["Copper"],
            "Alk_Phos": patient["Alk_Phos"],
            "SGOT": patient["SGOT"],
            "Tryglicerides": patient["Tryglicerides"],
            "Platelets": patient["Platelets"],
            "Prothrombin": patient["Prothrombin"]
        }


    # ======================================================
    # RUN MULTI-AGENT SYSTEM
    # ======================================================

    def predict(self, patient):

        print("\n")
        print("=" * 70)
        print("LIVERAI MULTI-AGENT PREDICTION")
        print("=" * 70)


        # ==================================================
        # SHARED KNOWLEDGE CONTEXT
        # ==================================================

        shared_context = {

            "patient": patient,

            "agents": {},

            "status": "running"
        }


        # ==================================================
        # 1. FATTY LIVER
        # ==================================================

        print("\n[1/3] Running Fatty Liver Agent...")

        try:

            fatty_data = self.prepare_fatty_data(
                patient
            )

            fatty_result = self.fatty_agent.predict(
                fatty_data
            )

            shared_context["agents"]["fatty_liver"] = (
                fatty_result
            )

            print("✓ Fatty Liver completed")

        except Exception as e:

            shared_context["agents"]["fatty_liver"] = {

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

            fibrosis_data = self.prepare_fibrosis_data(
                patient
            )

            fibrosis_result = self.fibrosis_agent.predict(
                fibrosis_data
            )

            shared_context["agents"]["fibrosis"] = (
                fibrosis_result
            )

            print("✓ Fibrosis completed")

        except Exception as e:

            shared_context["agents"]["fibrosis"] = {

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

            cirrhosis_data = self.prepare_cirrhosis_data(
                patient
            )

            cirrhosis_result = self.cirrhosis_agent.predict(
                cirrhosis_data
            )

            shared_context["agents"]["cirrhosis"] = (
                cirrhosis_result
            )

            print("✓ Cirrhosis completed")

        except Exception as e:

            shared_context["agents"]["cirrhosis"] = {

                "agent": "CirrhosisAgent",

                "status": "error",

                "error": str(e)
            }

            print("✗ Cirrhosis error:", e)


        # ==================================================
        # FINAL STATUS
        # ==================================================

        completed = 0

        for result in shared_context["agents"].values():

            if result.get("status") == "completed":

                completed += 1


        shared_context["agents_completed"] = completed

        shared_context["total_agents"] = 3

        shared_context["status"] = (
            "completed"
            if completed == 3
            else "partial"
        )


        return shared_context


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

def print_results(results):

    print("\n")
    print("=" * 70)
    print("LIVERAI MULTI-AGENT RESULTS")
    print("=" * 70)


    for name, result in results["agents"].items():

        print("\n" + name.upper())

        print(
            "Agent       :",
            result.get("agent")
        )

        print(
            "Model       :",
            result.get("model")
        )

        print(
            "Prediction  :",
            result.get("prediction")
        )

        print(
            "Probability :",
            result.get("probability")
        )

        print(
            "Status      :",
            result.get("status")
        )


    print("\n" + "-" * 70)

    print(
        "Agents completed:",
        f"{results['agents_completed']}/"
        f"{results['total_agents']}"
    )

    print(
        "Global status:",
        results["status"]
    )

    print("=" * 70)
