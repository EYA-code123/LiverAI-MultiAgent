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
from agents.clinical_reasoning_agent import ClinicalReasoningAgent


# ==========================================================
# LIVER AI ORCHESTRATOR
# ==========================================================

class LiverAIOrchestrator:

    def __init__(
        self,
        fatty_model_path,
        fibrosis_model_path,
        cirrhosis_model_path
    ):

        print("=" * 70)
        print("INITIALIZING LIVERAI MULTI-AGENT SYSTEM")
        print("=" * 70)

        # --------------------------------------------------
        # Check model paths
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

        cirrhosis_package = joblib.load(
            cirrhosis_model_path
        )

        # --------------------------------------------------
        # Initialize specialized agents
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

        # --------------------------------------------------
        # Initialize Clinical Reasoning Agent
        # --------------------------------------------------

        self.clinical_agent = ClinicalReasoningAgent()

        print("\n✓ Fatty Liver Agent initialized")
        print("✓ Fibrosis Agent initialized")
        print("✓ Cirrhosis Agent initialized")
        print("✓ Clinical Reasoning Agent initialized")

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

        # --------------------------------------------------
        # Shared Knowledge Context
        # --------------------------------------------------

        shared_context = {

            "patient": patient,

            "agents": {},

            "clinical_reasoning": None,

            "status": "running"
        }


        # ==================================================
        # 1. FATTY LIVER AGENT
        # ==================================================

        print("\n[1/4] Running Fatty Liver Agent...")

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

            print("✓ Fatty Liver Agent completed")

        except Exception as e:

            shared_context["agents"]["fatty_liver"] = {

                "agent": "FattyLiverAgent",

                "status": "error",

                "error": str(e)
            }

            print("✗ Fatty Liver Agent error:", e)


        # ==================================================
        # 2. FIBROSIS AGENT
        # ==================================================

        print("\n[2/4] Running Fibrosis Agent...")

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

            print("✓ Fibrosis Agent completed")

        except Exception as e:

            shared_context["agents"]["fibrosis"] = {

                "agent": "FibrosisAgent",

                "status": "error",

                "error": str(e)
            }

            print("✗ Fibrosis Agent error:", e)


        # ==================================================
        # 3. CIRRHOSIS AGENT
        # ==================================================

        print("\n[3/4] Running Cirrhosis Agent...")

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

            print("✓ Cirrhosis Agent completed")

        except Exception as e:

            shared_context["agents"]["cirrhosis"] = {

                "agent": "CirrhosisAgent",

                "status": "error",

                "error": str(e)
            }

            print("✗ Cirrhosis Agent error:", e)


        # ==================================================
        # 4. CLINICAL REASONING AGENT
        # ==================================================

        print("\n[4/4] Running Clinical Reasoning Agent...")

        try:

            clinical_result = self.clinical_agent.predict(
                shared_context["agents"]
            )

            shared_context["clinical_reasoning"] = (
                clinical_result
            )

            print("✓ Clinical Reasoning completed")

        except Exception as e:

            shared_context["clinical_reasoning"] = {

                "agent": "ClinicalReasoningAgent",

                "status": "error",

                "error": str(e)
            }

            print("✗ Clinical Reasoning error:", e)


        # ==================================================
        # FINAL STATUS
        # ==================================================

        completed = 0

        for result in shared_context["agents"].values():

            if result.get("status") == "completed":
                completed += 1


        clinical_status = shared_context[
            "clinical_reasoning"
        ].get("status")


        shared_context["agents_completed"] = completed

        shared_context["total_agents"] = 3

        shared_context["clinical_reasoning_completed"] = (
            clinical_status == "completed"
        )


        if (
            completed == 3
            and clinical_status == "completed"
        ):

            shared_context["status"] = "completed"

        else:

            shared_context["status"] = "partial"


        return shared_context


# ==========================================================
# DISPLAY FINAL RESULTS
# ==========================================================

def print_results(results):

    print("\n")
    print("=" * 70)
    print("LIVERAI FINAL RISK SUMMARY")
    print("=" * 70)


    # ======================================================
    # SPECIALIZED AGENTS
    # ======================================================

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


    # ======================================================
    # CLINICAL REASONING
    # ======================================================

    clinical = results.get(
        "clinical_reasoning",
        {}
    )

    print("\n")
    print("-" * 70)
    print("CLINICAL REASONING")
    print("-" * 70)

    print(
        "Overall Risk       :",
        clinical.get("overall_risk")
    )

    print(
        "Risk Score         :",
        clinical.get("risk_score")
    )

    print(
        "Confidence Level   :",
        clinical.get("confidence_level")
    )

    print(
        "Average Confidence:",
        clinical.get("average_confidence")
    )

    print(
        "Agents Used        :",
        clinical.get("agents_used")
    )


    # ======================================================
    # ABNORMAL FINDINGS
    # ======================================================

    print("\nAbnormal Findings:")

    for finding in clinical.get(
        "abnormal_findings",
        []
    ):

        print(" -", finding)


    # ======================================================
    # CLINICAL DECISION
    # ======================================================

    print("\nClinical Decision:")

    print(
        clinical.get(
            "clinical_decision",
            "Not available"
        )
    )


    # ======================================================
    # RECOMMENDATION
    # ======================================================

    print("\nRecommendation:")

    print(
        clinical.get(
            "recommendation",
            "Not available"
        )
    )


    # ======================================================
    # SYSTEM STATUS
    # ======================================================

    print("\n")
    print("=" * 70)

    print(
        "Specialized Agents:",
        f"{results['agents_completed']}/"
        f"{results['total_agents']}"
    )

    print(
        "Clinical Reasoning:",
        results[
            "clinical_reasoning_completed"
        ]
    )

    print(
        "Global Status:",
        results["status"]
    )

    print("=" * 70)
