"""
LiverAI - Multi-Agent Orchestrator
===================================

Coordinates the six specialized liver agents:

1. Fatty Liver Agent
2. Fibrosis Agent
3. Cirrhosis Agent
4. Tumor Classification Agent
5. Liver Segmentation Agent
6. Clinical Reasoning Agent

Important:
- Each agent receives only the input modality/features it expects.
- Missing modalities are handled gracefully.
- The orchestrator does NOT modify the individual agents.
"""

import os
import time
from typing import Any, Dict, Optional


# ============================================================
# AGENT IMPORTS
# ============================================================

from agents.fatty_liver_agent import FattyLiverAgent
from agents.fibrosis_agent import FibrosisAgent
from agents.cirrhosis_agent import CirrhosisAgent
from agents.tumor_classification_agent import TumorClassificationAgent
from agents.liver_segmentation_agent import LiverSegmentationAgent
from agents.clinical_reasoning_agent import ClinicalReasoningAgent


class LiverOrchestrator:
    """
    Central coordinator for the LiverAI multi-agent system.
    """

    # ========================================================
    # MODEL PATHS
    # ========================================================

    DEFAULT_PATHS = {
        "fatty_liver":
            "/content/drive/MyDrive/Fatty_Liver_Dataset/models/"
            "FattyLiver_LightGBM.pkl",

        "fibrosis":
            "/content/drive/MyDrive/Fibrosis Agent/"
            "XGBoost_model/xgboost_nafld.pkl",

        "cirrhosis":
            "/content/drive/MyDrive/.Cirrhosis Agent/"
            "XGBoost_model/XGBoost_Cirrhosis_fixed.joblib",

        "tumor":
            "/content/drive/MyDrive/models/tumor/"
            "efficientnet_b0_best.pth",

        "segmentation":
            "/content/drive/MyDrive/Liver Segmentation Agent/"
            "models/SegResNet3D_Liver_best.pth",

        "clinical_reasoning":
            "/content/drive/MyDrive/Clinical Reasoning Agent/"
            "tabtransformer_bupa",
    }

    # ========================================================
    # EXPECTED FEATURES
    # ========================================================

    FATTY_FEATURES = [
        "mcv",
        "alkphos",
        "sgpt",
        "sgot",
        "gammagt",
        "drinks",
    ]

    FIBROSIS_FEATURES = [
        "age",
        "male",
        "weight",
        "height",
        "bmi",
        "futime",
        "days",
        "test",
        "value",
    ]

    CIRRHOSIS_FEATURES = [
        "N_Days",
        "Status",
        "Drug",
        "Age",
        "Sex",
        "Ascites",
        "Hepatomegaly",
        "Spiders",
        "Edema",
        "Bilirubin",
        "Cholesterol",
        "Copper",
        "Albumin",
        "Alk_Phos",
        "SGOT",
        "Tryglicerides",
        "Platelets",
        "Prothrombin",
    ]

    CLINICAL_FEATURES = [
        "mcv",
        "alkphos",
        "sgpt",
        "sgot",
        "gammagt",
        "drinks",
    ]

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        model_paths: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
    ):
        self.name = "LiverOrchestrator"

        self.model_paths = self.DEFAULT_PATHS.copy()

        if model_paths is not None:
            self.model_paths.update(model_paths)

        self.device = device

        self.agents = {}
        self.agent_status = {}

        print("=" * 75)
        print("LIVERAI MULTI-AGENT ORCHESTRATOR")
        print("=" * 75)

        self._initialize_agents()

    # ========================================================
    # SAFE INITIALIZATION
    # ========================================================

    def _initialize_agents(self):
        """
        Initialize each agent independently.

        If one agent cannot be loaded, the other agents remain
        available.
        """

        # ----------------------------------------------------
        # FATTY LIVER
        # ----------------------------------------------------

        self._load_agent(
            name="fatty_liver",
            loader=lambda: FattyLiverAgent(
                self.model_paths["fatty_liver"]
            ),
        )

        # ----------------------------------------------------
        # FIBROSIS
        # ----------------------------------------------------

        self._load_agent(
            name="fibrosis",
            loader=lambda: FibrosisAgent(
                self.model_paths["fibrosis"]
            ),
        )

        # ----------------------------------------------------
        # CIRRHOSIS
        # ----------------------------------------------------

        self._load_agent(
            name="cirrhosis",
            loader=lambda: CirrhosisAgent(
                model_package=self.model_paths["cirrhosis"]
            ),
        )

        # ----------------------------------------------------
        # TUMOR CLASSIFICATION
        # ----------------------------------------------------

        self._load_agent(
            name="tumor",
            loader=lambda: TumorClassificationAgent(
                self.model_paths["tumor"]
            ),
        )

        # ----------------------------------------------------
        # LIVER SEGMENTATION
        # ----------------------------------------------------

        def load_segmentation():
            if self.device is not None:
                return LiverSegmentationAgent(
                    model_path=self.model_paths["segmentation"],
                    device=self.device,
                )

            return LiverSegmentationAgent(
                model_path=self.model_paths["segmentation"]
            )

        self._load_agent(
            name="segmentation",
            loader=load_segmentation,
        )

        # ----------------------------------------------------
        # CLINICAL REASONING
        # ----------------------------------------------------

        self._load_agent(
            name="clinical_reasoning",
            loader=lambda: ClinicalReasoningAgent(
                self.model_paths["clinical_reasoning"]
            ),
        )

        print("\n" + "=" * 75)
        print("AGENT INITIALIZATION SUMMARY")
        print("=" * 75)

        for name, status in self.agent_status.items():
            symbol = "✓" if status["status"] == "success" else "✗"
            print(f"{symbol} {name}: {status['status']}")

        print("=" * 75)

    # ========================================================
    # GENERIC AGENT LOADER
    # ========================================================

    def _load_agent(self, name: str, loader):
        """
        Safely load one agent.
        """

        try:
            print(f"\nLoading {name} agent...")

            agent = loader()

            self.agents[name] = agent

            self.agent_status[name] = {
                "status": "success",
                "error": None,
            }

            print(f"✓ {name} agent loaded")

        except Exception as exc:

            self.agent_status[name] = {
                "status": "error",
                "error": str(exc),
            }

            print(f"✗ {name} agent failed")
            print(f"  Error: {exc}")

    # ========================================================
    # FEATURE VALIDATION
    # ========================================================

    @staticmethod
    def _has_features(
        data: Optional[Dict[str, Any]],
        required_features,
    ) -> bool:

        if data is None:
            return False

        return all(
            feature in data
            for feature in required_features
        )

    # ========================================================
    # RUN ONE AGENT SAFELY
    # ========================================================

    def _run_agent(
        self,
        agent_name: str,
        input_data: Any,
    ) -> Dict[str, Any]:

        if agent_name not in self.agents:

            return {
                "status": "unavailable",
                "agent": agent_name,
                "error": "Agent is not initialized.",
            }

        if input_data is None:

            return {
                "status": "not_run",
                "agent": agent_name,
                "reason": "Required input was not provided.",
            }

        agent = self.agents[agent_name]

        start = time.time()

        try:

            if hasattr(agent, "run"):
                result = agent.run(input_data)

            elif hasattr(agent, "predict"):
                result = agent.predict(input_data)

            else:
                raise AttributeError(
                    f"{agent_name} has neither run() nor predict()."
                )

            if isinstance(result, dict):
                result.setdefault(
                    "agent",
                    agent_name,
                )

                result.setdefault(
                    "orchestrator_time",
                    time.time() - start,
                )

                return result

            return {
                "status": "success",
                "agent": agent_name,
                "result": result,
                "orchestrator_time": time.time() - start,
            }

        except Exception as exc:

            return {
                "status": "error",
                "agent": agent_name,
                "error": str(exc),
                "orchestrator_time": time.time() - start,
            }

    # ========================================================
    # FATTY LIVER
    # ========================================================

    def run_fatty_liver(
        self,
        patient_data: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not self._has_features(
            patient_data,
            self.FATTY_FEATURES,
        ):

            return {
                "status": "not_run",
                "agent": "fatty_liver",
                "reason": "Missing fatty liver features.",
                "required_features": self.FATTY_FEATURES,
            }

        data = {
            feature: patient_data[feature]
            for feature in self.FATTY_FEATURES
        }

        return self._run_agent(
            "fatty_liver",
            data,
        )

    # ========================================================
    # FIBROSIS
    # ========================================================

    def run_fibrosis(
        self,
        patient_data: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not self._has_features(
            patient_data,
            self.FIBROSIS_FEATURES,
        ):

            return {
                "status": "not_run",
                "agent": "fibrosis",
                "reason": "Missing fibrosis features.",
                "required_features": self.FIBROSIS_FEATURES,
            }

        data = {
            feature: patient_data[feature]
            for feature in self.FIBROSIS_FEATURES
        }

        return self._run_agent(
            "fibrosis",
            data,
        )

    # ========================================================
    # CIRRHOSIS
    # ========================================================

    def run_cirrhosis(
        self,
        patient_data: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not self._has_features(
            patient_data,
            self.CIRRHOSIS_FEATURES,
        ):

            return {
                "status": "not_run",
                "agent": "cirrhosis",
                "reason": "Missing cirrhosis features.",
                "required_features": self.CIRRHOSIS_FEATURES,
            }

        data = {
            feature: patient_data[feature]
            for feature in self.CIRRHOSIS_FEATURES
        }

        return self._run_agent(
            "cirrhosis",
            data,
        )

    # ========================================================
    # TUMOR CLASSIFICATION
    # ========================================================

    def run_tumor(
        self,
        image,
    ) -> Dict[str, Any]:

        return self._run_agent(
            "tumor",
            image,
        )

    # ========================================================
    # LIVER SEGMENTATION
    # ========================================================

    def run_segmentation(
        self,
        volume,
    ) -> Dict[str, Any]:

        return self._run_agent(
            "segmentation",
            volume,
        )

    # ========================================================
    # CLINICAL REASONING
    # ========================================================

    def run_clinical_reasoning(
        self,
        patient_data: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not self._has_features(
            patient_data,
            self.CLINICAL_FEATURES,
        ):

            return {
                "status": "not_run",
                "agent": "clinical_reasoning",
                "reason": "Missing clinical reasoning features.",
                "required_features": self.CLINICAL_FEATURES,
            }

        data = {
            feature: patient_data[feature]
            for feature in self.CLINICAL_FEATURES
        }

        return self._run_agent(
            "clinical_reasoning",
            data,
        )

    # ========================================================
    # COMPLETE ANALYSIS
    # ========================================================

    def analyze(
        self,
        patient_data: Optional[Dict[str, Any]] = None,
        tumor_image: Any = None,
        liver_volume: Any = None,
    ) -> Dict[str, Any]:

        start_time = time.time()

        if patient_data is None:
            patient_data = {}

        print("\n")
        print("=" * 75)
        print("LIVERAI - COMPLETE MULTI-AGENT ANALYSIS")
        print("=" * 75)

        # ----------------------------------------------------
        # RUN SPECIALIZED AGENTS
        # ----------------------------------------------------

        fatty_result = self.run_fatty_liver(
            patient_data
        )

        fibrosis_result = self.run_fibrosis(
            patient_data
        )

        cirrhosis_result = self.run_cirrhosis(
            patient_data
        )

        tumor_result = self.run_tumor(
            tumor_image
        )

        segmentation_result = self.run_segmentation(
            liver_volume
        )

        clinical_result = self.run_clinical_reasoning(
            patient_data
        )

        # ----------------------------------------------------
        # COLLECT RESULTS
        # ----------------------------------------------------

        results = {
            "fatty_liver": fatty_result,
            "fibrosis": fibrosis_result,
            "cirrhosis": cirrhosis_result,
            "tumor": tumor_result,
            "segmentation": segmentation_result,
            "clinical_reasoning": clinical_result,
        }

        # ----------------------------------------------------
        # STATUS SUMMARY
        # ----------------------------------------------------

        status_summary = {}

        for name, result in results.items():

            status_summary[name] = result.get(
                "status",
                "unknown",
            )

        # ----------------------------------------------------
        # FINAL REPORT
        # ----------------------------------------------------

        report = {
            "status": "success",
            "orchestrator": self.name,

            "agents": results,

            "agent_status": status_summary,

            "inputs": {
                "patient_data_available": bool(patient_data),
                "tumor_image_available": tumor_image is not None,
                "liver_volume_available": liver_volume is not None,
            },

            "total_inference_time":
                time.time() - start_time,
        }

        print("\n" + "=" * 75)
        print("MULTI-AGENT ANALYSIS SUMMARY")
        print("=" * 75)

        for name, result in results.items():

            status = result.get(
                "status",
                "unknown",
            )

            print(f"{name:20s}: {status}")

        print("=" * 75)

        return report

    # ========================================================
    # TEST
    # ========================================================

    def test(self):

        print("\n")
        print("=" * 75)
        print("LIVER ORCHESTRATOR - TECHNICAL TEST")
        print("=" * 75)

        # ----------------------------------------------------
        # Test data for agents that share compatible features
        # ----------------------------------------------------

        test_patient = {

            # Fatty Liver / Clinical Reasoning
            "mcv": 85.0,
            "alkphos": 85.0,
            "sgpt": 45.0,
            "sgot": 35.0,
            "gammagt": 50.0,
            "drinks": 5.0,

            # Fibrosis
            "age": 50.0,
            "male": 1.0,
            "weight": 75.0,
            "height": 170.0,
            "bmi": 26.0,
            "futime": 1000.0,
            "days": 30.0,
            "test": 1.0,
            "value": 10.0,

            # Cirrhosis
            "N_Days": 1000,
            "Status": "C",
            "Drug": "D-penicillamine",
            "Age": 50,
            "Sex": "M",
            "Ascites": "N",
            "Hepatomegaly": "N",
            "Spiders": "N",
            "Edema": "N",
            "Bilirubin": 1.0,
            "Cholesterol": 200.0,
            "Copper": 100.0,
            "Albumin": 3.5,
            "Alk_Phos": 1000.0,
            "SGOT": 100.0,
            "Tryglicerides": 100.0,
            "Platelets": 200.0,
            "Prothrombin": 10.0,
        }

        report = self.analyze(
            patient_data=test_patient,
            tumor_image=None,
            liver_volume=None,
        )

        print("\n")
        print("=" * 75)
        print("ORCHESTRATOR TEST RESULTS")
        print("=" * 75)

        for name, result in report["agents"].items():

            print(
                f"\n{name.upper()}"
            )

            print(
                f"Status: "
                f"{result.get('status')}"
            )

            if "prediction" in result:
                print(
                    f"Prediction: "
                    f"{result['prediction']}"
                )

            if "confidence" in result:
                print(
                    f"Confidence: "
                    f"{result['confidence']}"
                )

        print("\n" + "=" * 75)

        return report


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    orchestrator = LiverOrchestrator()

    orchestrator.test()
