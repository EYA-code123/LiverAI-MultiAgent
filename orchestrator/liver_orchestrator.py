# =============================================================================
# LiverAI-MultiAgent
# COMPLETE MULTI-AGENT ORCHESTRATOR
# =============================================================================

from __future__ import annotations

import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# OPTIONAL COORDINATION MODULES
# =============================================================================

try:
    from orchestrator.schemas import AgentResult
except Exception:
    AgentResult = None


try:
    from coordinator.trust import TrustManager
except Exception:
    TrustManager = None


try:
    from coordinator.adaptive_fusion import AdaptiveFusion
except Exception:
    AdaptiveFusion = None


try:
    from coordinator.conflict import ConflictDetector
except Exception:
    ConflictDetector = None


try:
    from coordinator.decision import DecisionEngine
except Exception:
    DecisionEngine = None


# =============================================================================
# AGENTS
# =============================================================================

try:
    from agents.fatty_liver_agent import FattyLiverAgent
except Exception:
    FattyLiverAgent = None


try:
    from agents.fibrosis_agent import FibrosisAgent
except Exception:
    FibrosisAgent = None


try:
    from agents.cirrhosis_agent import CirrhosisAgent
except Exception:
    CirrhosisAgent = None


try:
    from agents.tumor_classification_agent import TumorClassificationAgent
except Exception:
    TumorClassificationAgent = None


try:
    from agents.liver_segmentation_agent import LiverSegmentationAgent
except Exception:
    LiverSegmentationAgent = None


try:
    from agents.clinical_reasoning_agent import ClinicalReasoningAgent
except Exception:
    ClinicalReasoningAgent = None


# =============================================================================
# DEFAULT MODEL PATHS
# =============================================================================

DEFAULT_MODEL_PATHS = {

    "fatty_liver":
        "/content/drive/MyDrive/"
        "Fatty_Liver_Dataset/models/FattyLiver_LightGBM.pkl",

    "fibrosis":
        "/content/drive/MyDrive/"
        "Fibrosis Agent/XGBoost_model/xgboost_nafld.pkl",

    "cirrhosis":
        "/content/drive/MyDrive/"
        ".Cirrhosis Agent/XGBoost_model/"
        "XGBoost_Cirrhosis_fixed.joblib",

    "tumor":
        "/content/drive/MyDrive/"
        "models/tumor/efficientnet_b0_best.pth",

    "segmentation":
        "/content/drive/MyDrive/"
        "Liver Segmentation Agent/models/"
        "SegResNet3D_Liver_best.pth",

    "clinical":
        "/content/drive/MyDrive/"
        "Clinical Reasoning Agent/tabtransformer_bupa",
}


# =============================================================================
# EXPECTED FEATURES
# =============================================================================

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


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class LiverAIOrchestrator:
    """
    Central coordinator for the LiverAI multi-agent system.

    Agents:

        1. Fatty Liver
        2. Fibrosis
        3. Cirrhosis
        4. Tumor Classification
        5. Liver Segmentation
        6. Clinical Reasoning

    Important:
        Each agent receives ONLY the modality/features it was trained for.
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        cirrhosis_agent=None,
        fatty_liver_agent=None,
        clinical_agent=None,
        fibrosis_agent=None,
        tumor_agent=None,
        segmentation_agent=None,
        model_paths: Optional[Dict[str, str]] = None,
        auto_load=True,
        device=None,
    ):

        self.name = "LiverAI Multi-Agent Orchestrator"

        self.model_paths = dict(DEFAULT_MODEL_PATHS)

        if model_paths is not None:
            self.model_paths.update(model_paths)

        self.device = device

        # ---------------------------------------------------------------------
        # User-supplied agents
        # ---------------------------------------------------------------------

        self.cirrhosis_agent = cirrhosis_agent
        self.fatty_liver_agent = fatty_liver_agent
        self.fibrosis_agent = fibrosis_agent
        self.tumor_agent = tumor_agent
        self.segmentation_agent = segmentation_agent
        self.clinical_agent = clinical_agent

        # ---------------------------------------------------------------------
        # Automatically load missing agents
        # ---------------------------------------------------------------------

        if auto_load:

            self._load_missing_agents()

        # ---------------------------------------------------------------------
        # Registry
        # ---------------------------------------------------------------------

        self.agents = {

            "fatty_liver":
                self.fatty_liver_agent,

            "fibrosis":
                self.fibrosis_agent,

            "cirrhosis":
                self.cirrhosis_agent,

            "tumor_classification":
                self.tumor_agent,

            "liver_segmentation":
                self.segmentation_agent,

            "clinical_reasoning":
                self.clinical_agent,
        }

        # ---------------------------------------------------------------------
        # Coordination modules
        # ---------------------------------------------------------------------

        self.trust_manager = (
            TrustManager()
            if TrustManager is not None
            else None
        )

        self.adaptive_fusion = (
            AdaptiveFusion()
            if AdaptiveFusion is not None
            else None
        )

        self.conflict_detector = (
            ConflictDetector()
            if ConflictDetector is not None
            else None
        )

        self.decision_engine = (
            DecisionEngine()
            if DecisionEngine is not None
            else None
        )

        # ---------------------------------------------------------------------
        # State
        # ---------------------------------------------------------------------

        self.last_results = {}
        self.last_assessment = None
        self.execution_log = []

        # ---------------------------------------------------------------------
        # Display
        # ---------------------------------------------------------------------

        self._print_system_status()

    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def _load_missing_agents(self):

        self._log(
            "\n"
            + "=" * 80
        )

        self._log(
            "LOADING LIVERAI AGENTS"
        )

        self._log(
            "=" * 80
        )

        # =====================================================================
        # 1. FATTY LIVER
        # =====================================================================

        if self.fatty_liver_agent is None:

            path = self.model_paths["fatty_liver"]

            try:

                if not os.path.exists(path):
                    raise FileNotFoundError(path)

                import joblib

                model_package = joblib.load(path)

                self.fatty_liver_agent = (
                    FattyLiverAgent(
                        model_package
                    )
                )

                self._log(
                    "✓ FattyLiverAgent loaded"
                )

            except Exception as e:

                self._log(
                    f"✗ FattyLiverAgent loading failed: {e}"
                )

                self.fatty_liver_agent = None

        # =====================================================================
        # 2. FIBROSIS
        # =====================================================================

        if self.fibrosis_agent is None:

            path = self.model_paths["fibrosis"]

            try:

                if not os.path.exists(path):
                    raise FileNotFoundError(path)

                import joblib

                fibrosis_model = joblib.load(path)

                self.fibrosis_agent = (
                    FibrosisAgent(
                        fibrosis_model
                    )
                )

                self._log(
                    "✓ FibrosisAgent loaded"
                )

            except Exception as e:

                self._log(
                    f"✗ FibrosisAgent loading failed: {e}"
                )

                self.fibrosis_agent = None

        # =====================================================================
        # 3. CIRRHOSIS
        # =====================================================================

        if self.cirrhosis_agent is None:

            path = self.model_paths["cirrhosis"]

            try:

                if not os.path.exists(path):
                    raise FileNotFoundError(path)

                self.cirrhosis_agent = (
                    CirrhosisAgent(
                        path
                    )
                )

                self._log(
                    "✓ CirrhosisAgent loaded"
                )

            except Exception as e:

                self._log(
                    f"✗ CirrhosisAgent loading failed: {e}"
                )

                self.cirrhosis_agent = None

        # =====================================================================
        # 4. TUMOR
        # =====================================================================

        if self.tumor_agent is None:

            path = self.model_paths["tumor"]

            try:

                if not os.path.exists(path):
                    raise FileNotFoundError(path)

                self.tumor_agent = (
                    TumorClassificationAgent(
                        path
                    )
                )

                self._log(
                    "✓ TumorClassificationAgent loaded"
                )

            except Exception as e:

                self._log(
                    f"✗ TumorClassificationAgent loading failed: {e}"
                )

                self.tumor_agent = None

        # =====================================================================
        # 5. SEGMENTATION
        # =====================================================================

        if self.segmentation_agent is None:

            path = self.model_paths["segmentation"]

            try:

                if not os.path.exists(path):
                    raise FileNotFoundError(path)

                self.segmentation_agent = (
                    LiverSegmentationAgent(
                        model_path=path,
                        device=self.device,
                        target_size=(128, 128, 64),
                        threshold=0.5,
                    )
                )

                self._log(
                    "✓ LiverSegmentationAgent loaded"
                )

            except Exception as e:

                self._log(
                    f"✗ LiverSegmentationAgent loading failed: {e}"
                )

                self.segmentation_agent = None

        # =====================================================================
        # 6. CLINICAL REASONING
        # =====================================================================

        if self.clinical_agent is None:

            path = self.model_paths["clinical"]

            try:

                if not os.path.isdir(path):
                    raise FileNotFoundError(path)

                self.clinical_agent = (
                    ClinicalReasoningAgent(
                        path
                    )
                )

                self._log(
                    "✓ ClinicalReasoningAgent loaded"
                )

            except Exception as e:

                self._log(
                    f"✗ ClinicalReasoningAgent loading failed: {e}"
                )

                self.clinical_agent = None

        self._log(
            "=" * 80
        )

    # =========================================================================
    # SYSTEM STATUS
    # =========================================================================

    def _print_system_status(self):

        print("\n")
        print("=" * 80)
        print("LIVERAI MULTI-AGENT ORCHESTRATOR")
        print("=" * 80)

        print("\nRegistered Agents:")

        for name, agent in self.agents.items():

            status = (
                "READY"
                if agent is not None
                else "NOT LOADED"
            )

            print(
                f"  {name:<25} : {status}"
            )

        print("\nCoordination Modules:")

        print(
            f"  Trust Manager             : "
            f"{'READY' if self.trust_manager else 'NOT AVAILABLE'}"
        )

        print(
            f"  Adaptive Fusion           : "
            f"{'READY' if self.adaptive_fusion else 'NOT AVAILABLE'}"
        )

        print(
            f"  Conflict Detector         : "
            f"{'READY' if self.conflict_detector else 'NOT AVAILABLE'}"
        )

        print(
            f"  Decision Engine           : "
            f"{'READY' if self.decision_engine else 'NOT AVAILABLE'}"
        )

        print("=" * 80)

    # =========================================================================
    # LOGGING
    # =========================================================================

    def _log(self, message):

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        self.execution_log.append(
            {
                "timestamp": timestamp,
                "message": message,
            }
        )

        print(message)

    # =========================================================================
    # INPUT ADAPTERS
    # =========================================================================

    @staticmethod
    def _prepare_dict(
        data,
        expected_features
    ):

        if data is None:
            return None

        if isinstance(data, pd.DataFrame):

            if len(data) == 0:
                return None

            data = data.iloc[0].to_dict()

        elif not isinstance(data, dict):

            raise TypeError(
                "Input must be a dictionary or pandas DataFrame."
            )

        prepared = {}

        for feature in expected_features:

            if feature in data:

                prepared[feature] = data[feature]

            else:

                prepared[feature] = np.nan

        return prepared

    # =========================================================================
    # MODALITY-SPECIFIC ADAPTERS
    # =========================================================================

    def _prepare_fatty_input(
        self,
        clinical_data
    ):

        return self._prepare_dict(
            clinical_data,
            FATTY_FEATURES
        )

    def _prepare_clinical_input(
        self,
        clinical_data
    ):

        return self._prepare_dict(
            clinical_data,
            FATTY_FEATURES
        )

    def _prepare_fibrosis_input(
        self,
        fibrosis_input
    ):

        return self._prepare_dict(
            fibrosis_input,
            FIBROSIS_FEATURES
        )

    def _prepare_cirrhosis_input(
        self,
        cirrhosis_input
    ):

        return self._prepare_dict(
            cirrhosis_input,
            CIRRHOSIS_FEATURES
        )

    # =========================================================================
    # SAFE EXECUTION
    # =========================================================================

    def _execute_agent(
        self,
        agent_name,
        agent,
        input_data,
        task_type,
    ):

        start_time = time.perf_counter()

        self._log(
            f"\n[{agent_name}] START"
        )

        # ---------------------------------------------------------------------
        # Agent unavailable
        # ---------------------------------------------------------------------

        if agent is None:

            self._log(
                f"[{agent_name}] NOT AVAILABLE"
            )

            return self._empty_result(
                agent_name,
                task_type,
                "Agent not available."
            )

        # ---------------------------------------------------------------------
        # No input
        # ---------------------------------------------------------------------

        if input_data is None:

            self._log(
                f"[{agent_name}] NO INPUT → SKIPPED"
            )

            return self._empty_result(
                agent_name,
                task_type,
                "Required input not provided."
            )

        # ---------------------------------------------------------------------
        # Execute
        # ---------------------------------------------------------------------

        try:

            if hasattr(agent, "predict"):

                result = agent.predict(
                    input_data
                )

            elif hasattr(agent, "run"):

                result = agent.run(
                    input_data
                )

            elif hasattr(agent, "analyze"):

                result = agent.analyze(
                    input_data
                )

            else:

                raise AttributeError(
                    f"{agent_name} does not provide "
                    "predict(), run(), or analyze()."
                )

            # -----------------------------------------------------------------
            # Normalize
            # -----------------------------------------------------------------

            if result is None:

                result = {}

            if not isinstance(result, dict):

                result = {
                    "prediction": result
                }

            elapsed_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

            result.setdefault(
                "agent_id",
                agent_name
            )

            result.setdefault(
                "agent",
                agent_name
            )

            result.setdefault(
                "task_type",
                task_type
            )

            result.setdefault(
                "status",
                "completed"
            )

            result.setdefault(
                "prediction",
                None
            )

            result.setdefault(
                "probability",
                None
            )

            result.setdefault(
                "confidence",
                0.0
            )

            result.setdefault(
                "uncertainty",
                1.0 -
                self._clip(
                    result.get(
                        "confidence",
                        0.0
                    )
                )
            )

            result.setdefault(
                "quality",
                1.0
            )

            result.setdefault(
                "missing_data_ratio",
                0.0
            )

            result.setdefault(
                "latency_ms",
                elapsed_ms
            )

            result.setdefault(
                "details",
                {}
            )

            result.setdefault(
                "explanation",
                None
            )

            result.setdefault(
                "error",
                None
            )

            # -----------------------------------------------------------------
            # Safe numerical values
            # -----------------------------------------------------------------

            result["confidence"] = self._clip(
                result.get(
                    "confidence",
                    0.0
                )
            )

            result["uncertainty"] = self._clip(
                result.get(
                    "uncertainty",
                    1.0
                )
            )

            result["quality"] = self._clip(
                result.get(
                    "quality",
                    0.0
                )
            )

            result["missing_data_ratio"] = self._clip(
                result.get(
                    "missing_data_ratio",
                    0.0
                )
            )

            try:

                result["latency_ms"] = max(
                    0.0,
                    float(
                        result.get(
                            "latency_ms",
                            elapsed_ms
                        )
                    )
                )

            except Exception:

                result["latency_ms"] = elapsed_ms

            self._log(
                f"[{agent_name}] COMPLETED "
                f"| confidence="
                f"{result['confidence']:.3f}"
            )

            return result

        except Exception as e:

            elapsed_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

            self._log(
                f"[{agent_name}] ERROR: {e}"
            )

            traceback.print_exc()

            return {
                "agent_id":
                    agent_name,

                "agent":
                    agent_name,

                "task_type":
                    task_type,

                "status":
                    "error",

                "prediction":
                    None,

                "probability":
                    None,

                "confidence":
                    0.0,

                "uncertainty":
                    1.0,

                "quality":
                    0.0,

                "missing_data_ratio":
                    1.0,

                "latency_ms":
                    elapsed_ms,

                "details":
                    {},

                "explanation":
                    None,

                "error":
                    str(e),

                "traceback":
                    traceback.format_exc(),
            }

    # =========================================================================
    # EMPTY RESULT
    # =========================================================================

    @staticmethod
    def _empty_result(
        agent_name,
        task_type,
        error
    ):

        return {
            "agent_id":
                agent_name,

            "agent":
                agent_name,

            "task_type":
                task_type,

            "status":
                "not_available",

            "prediction":
                None,

            "probability":
                None,

            "confidence":
                0.0,

            "uncertainty":
                1.0,

            "quality":
                0.0,

            "missing_data_ratio":
                1.0,

            "latency_ms":
                0.0,

            "details":
                {},

            "explanation":
                None,

            "error":
                error,
        }

    # =========================================================================
    # INDIVIDUAL AGENTS
    # =========================================================================

    def run_fatty_liver(
        self,
        clinical_data
    ):

        prepared_data = (
            self._prepare_fatty_input(
                clinical_data
            )
        )

        return self._execute_agent(
            "FattyLiverAgent",
            self.fatty_liver_agent,
            prepared_data,
            "fatty_liver_classification",
        )

    # -------------------------------------------------------------------------

    def run_fibrosis(
        self,
        fibrosis_input
    ):

        prepared_data = (
            self._prepare_fibrosis_input(
                fibrosis_input
            )
        )

        return self._execute_agent(
            "FibrosisAgent",
            self.fibrosis_agent,
            prepared_data,
            "fibrosis_classification",
        )

    # -------------------------------------------------------------------------

    def run_cirrhosis(
        self,
        cirrhosis_input
    ):

        prepared_data = (
            self._prepare_cirrhosis_input(
                cirrhosis_input
            )
        )

        return self._execute_agent(
            "CirrhosisAgent",
            self.cirrhosis_agent,
            prepared_data,
            "cirrhosis_classification",
        )

    # -------------------------------------------------------------------------

    def run_tumor_classification(
        self,
        image
    ):

        return self._execute_agent(
            "TumorClassificationAgent",
            self.tumor_agent,
            image,
            "tumor_classification",
        )

    # -------------------------------------------------------------------------

    def run_liver_segmentation(
        self,
        volume
    ):

        return self._execute_agent(
            "LiverSegmentationAgent",
            self.segmentation_agent,
            volume,
            "liver_segmentation",
        )

    # =========================================================================
    # SPECIALIZED AGENTS
    # =========================================================================

    def run_specialized_agents(
        self,
        clinical_data=None,
        fibrosis_input=None,
        cirrhosis_input=None,
        image=None,
        volume=None,
    ):

        self._log(
            "\n"
            + "=" * 80
        )

        self._log(
            "RUNNING SPECIALIZED AGENTS"
        )

        self._log(
            "=" * 80
        )

        results = {}

        # ---------------------------------------------------------------------
        # 1. FATTY LIVER
        # ---------------------------------------------------------------------

        results["fatty_liver"] = (
            self.run_fatty_liver(
                clinical_data
            )
        )

        # ---------------------------------------------------------------------
        # 2. FIBROSIS
        # ---------------------------------------------------------------------

        results["fibrosis"] = (
            self.run_fibrosis(
                fibrosis_input
            )
        )

        # ---------------------------------------------------------------------
        # 3. CIRRHOSIS
        # ---------------------------------------------------------------------

        results["cirrhosis"] = (
            self.run_cirrhosis(
                cirrhosis_input
            )
        )

        # ---------------------------------------------------------------------
        # 4. TUMOR
        # ---------------------------------------------------------------------

        results["tumor_classification"] = (
            self.run_tumor_classification(
                image
            )
        )

        # ---------------------------------------------------------------------
        # 5. SEGMENTATION
        # ---------------------------------------------------------------------

        results["liver_segmentation"] = (
            self.run_liver_segmentation(
                volume
            )
        )

        self.last_results = results

        return results

    # =========================================================================
    # STANDARDIZATION + TRUST
    # =========================================================================

    def _to_agent_results(
        self,
        raw_results
    ):

        if AgentResult is None:

            return raw_results

        agent_results = []

        for result in raw_results.values():

            try:

                agent_result = (
                    AgentResult.from_dict(
                        result
                    )
                )

            except Exception as e:

                self._log(
                    f"AgentResult conversion failed: {e}"
                )

                continue

            # -----------------------------------------------------------------
            # TRUST
            # -----------------------------------------------------------------

            if self.trust_manager is not None:

                try:

                    trust = (
                        self.trust_manager.compute_trust(
                            agent_id=
                                agent_result.agent_id,

                            confidence=
                                agent_result.confidence,

                            quality=
                                agent_result.quality,

                            uncertainty=
                                agent_result.uncertainty,

                            missing_data_ratio=
                                agent_result.missing_data_ratio,
                        )
                    )

                    agent_result.trust = self._clip(
                        trust
                    )

                except Exception:

                    agent_result.trust = (
                        agent_result.confidence
                    )

            else:

                agent_result.trust = (
                    agent_result.confidence
                )

            agent_results.append(
                agent_result
            )

        return agent_results

    # =========================================================================
    # FALLBACK FUSION
    # =========================================================================

    def _fallback_fusion(
        self,
        agent_results
    ):

        evidence = []

        for result in agent_results:

            status = getattr(
                result,
                "status",
                None
            )

            if status not in [
                "success",
                "completed",
            ]:
                continue

            prediction = getattr(
                result,
                "prediction",
                None
            )

            if prediction is None:
                continue

            trust = getattr(
                result,
                "trust",
                None
            )

            evidence.append(
                {
                    "agent_id":
                        getattr(
                            result,
                            "agent_id",
                            None
                        ),

                    "task_type":
                        getattr(
                            result,
                            "task_type",
                            None
                        ),

                    "prediction":
                        prediction,

                    "probability":
                        getattr(
                            result,
                            "probability",
                            None
                        ),

                    "confidence":
                        getattr(
                            result,
                            "confidence",
                            0.0
                        ),

                    "uncertainty":
                        getattr(
                            result,
                            "uncertainty",
                            1.0
                        ),

                    "quality":
                        getattr(
                            result,
                            "quality",
                            0.0
                        ),

                    "trust":
                        trust,
                }
            )

        return {
            "status":
                "success",

            "evidence":
                evidence,

            "num_valid_agents":
                len(evidence),

            "weights":
                {
                    item["agent_id"]:
                        (
                            item["trust"]
                            if item["trust"] is not None
                            else 0.0
                        )
                    for item in evidence
                },
        }

    # =========================================================================
    # FALLBACK CONFLICT DETECTION
    # =========================================================================

    def _fallback_conflicts(
        self,
        agent_results
    ):

        conflicts = []

        task_groups = {}

        for result in agent_results:

            task_type = getattr(
                result,
                "task_type",
                None
            )

            task_groups.setdefault(
                task_type,
                []
            ).append(
                result
            )

        for task_type, group in task_groups.items():

            predictions = [
                str(
                    getattr(
                        item,
                        "prediction",
                        None
                    )
                )
                for item in group
                if getattr(
                    item,
                    "prediction",
                    None
                ) is not None
            ]

            # Only meaningful if there are multiple agents
            # performing the same task.

            if len(group) > 1:

                if len(
                    set(predictions)
                ) > 1:

                    conflicts.append(
                        {
                            "task_type":
                                task_type,

                            "predictions":
                                predictions,

                            "agents":
                                [
                                    getattr(
                                        item,
                                        "agent_id",
                                        None
                                    )
                                    for item in group
                                ],
                        }
                    )

        return conflicts

    # =========================================================================
    # CLINICAL REASONING
    # =========================================================================

    def run_clinical_reasoning(
        self,
        clinical_data,
        agent_results=None,
        fusion_result=None,
        conflicts=None,
    ):

        self._log(
            "\n"
            + "=" * 80
        )

        self._log(
            "CLINICAL REASONING AGENT"
        )

        self._log(
            "=" * 80
        )

        # ---------------------------------------------------------------------
        # Agent unavailable
        # ---------------------------------------------------------------------

        if self.clinical_agent is None:

            return self._empty_result(
                "ClinicalReasoningAgent",
                "clinical_reasoning",
                "Clinical reasoning agent not available."
            )

        # ---------------------------------------------------------------------
        # IMPORTANT
        #
        # ClinicalReasoningAgent was trained on the BUPA six-feature input.
        #
        # We DO NOT send:
        #     agents
        #     fusion
        #     conflicts
        #
        # to the trained model.
        #
        # The other agent results remain available to the orchestrator.
        # ---------------------------------------------------------------------

        prepared_data = (
            self._prepare_clinical_input(
                clinical_data
            )
        )

        if prepared_data is None:

            return self._empty_result(
                "ClinicalReasoningAgent",
                "clinical_reasoning",
                "Clinical reasoning input not provided."
            )

        try:

            start_time = time.perf_counter()

            if hasattr(
                self.clinical_agent,
                "predict"
            ):

                result = (
                    self.clinical_agent.predict(
                        prepared_data
                    )
                )

            elif hasattr(
                self.clinical_agent,
                "run"
            ):

                result = (
                    self.clinical_agent.run(
                        prepared_data
                    )
                )

            elif hasattr(
                self.clinical_agent,
                "analyze"
            ):

                result = (
                    self.clinical_agent.analyze(
                        prepared_data
                    )
                )

            else:

                raise AttributeError(
                    "ClinicalReasoningAgent does not provide "
                    "predict(), run(), or analyze()."
                )

            if result is None:

                result = {}

            if not isinstance(
                result,
                dict
            ):

                result = {
                    "prediction":
                        result
                }

            elapsed_ms = (
                time.perf_counter()
                -
                start_time
            ) * 1000.0

            result.setdefault(
                "agent_id",
                "ClinicalReasoningAgent"
            )

            result.setdefault(
                "agent",
                "ClinicalReasoningAgent"
            )

            result.setdefault(
                "task_type",
                "clinical_reasoning"
            )

            result.setdefault(
                "status",
                "completed"
            )

            result.setdefault(
                "confidence",
                0.0
            )

            result.setdefault(
                "uncertainty",
                1.0 -
                self._clip(
                    result.get(
                        "confidence",
                        0.0
                    )
                )
            )

            result.setdefault(
                "quality",
                1.0
            )

            result.setdefault(
                "missing_data_ratio",
                0.0
            )

            result.setdefault(
                "latency_ms",
                elapsed_ms
            )

            result.setdefault(
                "error",
                None
            )

            result["confidence"] = self._clip(
                result.get(
                    "confidence",
                    0.0
                )
            )

            result["uncertainty"] = self._clip(
                result.get(
                    "uncertainty",
                    1.0
                )
            )

            result["quality"] = self._clip(
                result.get(
                    "quality",
                    0.0
                )
            )

            result["missing_data_ratio"] = self._clip(
                result.get(
                    "missing_data_ratio",
                    0.0
                )
            )

            self._log(
                "✓ Clinical reasoning completed"
            )

            return result

        except Exception as e:

            self._log(
                f"✗ Clinical reasoning error: {e}"
            )

            traceback.print_exc()

            return {
                "agent_id":
                    "ClinicalReasoningAgent",

                "agent":
                    "ClinicalReasoningAgent",

                "task_type":
                    "clinical_reasoning",

                "status":
                    "error",

                "prediction":
                    None,

                "confidence":
                    0.0,

                "uncertainty":
                    1.0,

                "quality":
                    0.0,

                "missing_data_ratio":
                    1.0,

                "latency_ms":
                    0.0,

                "error":
                    str(e),

                "traceback":
                    traceback.format_exc(),
            }

    # =========================================================================
    # FALLBACK DECISION
    # =========================================================================

    def _fallback_decision(
        self,
        agent_results,
        conflicts,
        clinical_reasoning
    ):

        valid = [

            result

            for result in agent_results

            if getattr(
                result,
                "status",
                None
            ) in [
                "success",
                "completed",
            ]
        ]

        if not valid:

            return {
                "status":
                    "insufficient_evidence",

                "risk_level":
                    "unknown",

                "risk_score":
                    0.0,

                "decision_confidence":
                    0.0,

                "request_additional_tests":
                    True,

                "note":
                    "No valid specialized-agent result available."
            }

        confidences = [

            self._clip(
                getattr(
                    result,
                    "confidence",
                    0.0
                )
            )

            for result in valid
        ]

        average_confidence = (
            sum(confidences)
            /
            max(
                len(confidences),
                1
            )
        )

        return {
            "status":
                "completed",

            "risk_level":
                "not_computed",

            "risk_score":
                0.0,

            "decision_confidence":
                float(
                    average_confidence
                ),

            "request_additional_tests":
                bool(
                    len(conflicts) > 0
                ),

            "conflicts_detected":
                len(conflicts),

            "note":
                (
                    "System-level aggregation only. "
                    "No medical diagnosis is produced."
                ),
        }

    # =========================================================================
    # COMPLETE PIPELINE
    # =========================================================================

    def run(
        self,
        patient_id,
        clinical_data=None,
        fibrosis_input=None,
        cirrhosis_input=None,
        image=None,
        volume=None,
    ):

        self.execution_log = []

        self._log(
            "\n"
            + "=" * 80
        )

        self._log(
            f"LIVERAI ANALYSIS — PATIENT {patient_id}"
        )

        self._log(
            "=" * 80
        )

        # =========================================================================
        # STEP 1
        # =========================================================================

        self._log(
            "\nSTEP 1/6 → Specialized Agents"
        )

        raw_results = (
            self.run_specialized_agents(
                clinical_data=
                    clinical_data,

                fibrosis_input=
                    fibrosis_input,

                cirrhosis_input=
                    cirrhosis_input,

                image=
                    image,

                volume=
                    volume,
            )
        )

        # =========================================================================
        # STEP 2
        # =========================================================================

        self._log(
            "\nSTEP 2/6 → Standardization + Trust"
        )

        agent_results = (
            self._to_agent_results(
                raw_results
            )
        )

        # If AgentResult is unavailable, use empty fusion
        # instead of crashing.

        if AgentResult is None:

            fusion_result = (
                {
                    "status":
                        "success",

                    "evidence":
                        raw_results
                }
            )

            conflicts = []

        else:

            # =====================================================================
            # STEP 3
            # =====================================================================

            self._log(
                "\nSTEP 3/6 → Adaptive Evidence Fusion"
            )

            if self.adaptive_fusion is not None:

                try:

                    fusion_result = (
                        self.adaptive_fusion.fuse(
                            agent_results
                        )
                    )

                except Exception as e:

                    self._log(
                        f"Fusion module failed: {e}"
                    )

                    fusion_result = (
                        self._fallback_fusion(
                            agent_results
                        )
                    )

            else:

                fusion_result = (
                    self._fallback_fusion(
                        agent_results
                    )
                )

            # =====================================================================
            # STEP 4
            # =====================================================================

            self._log(
                "\nSTEP 4/6 → Conflict Detection"
            )

            if self.conflict_detector is not None:

                try:

                    conflicts = (
                        self.conflict_detector.detect(
                            agent_results
                        )
                    )

                except Exception as e:

                    self._log(
                        f"Conflict detector failed: {e}"
                    )

                    conflicts = (
                        self._fallback_conflicts(
                            agent_results
                        )
                    )

            else:

                conflicts = (
                    self._fallback_conflicts(
                        agent_results
                    )
                )

        # =========================================================================
        # STEP 5
        # =========================================================================

        self._log(
            "\nSTEP 5/6 → Clinical Reasoning"
        )

        clinical_result = (
            self.run_clinical_reasoning(
                clinical_data=
                    clinical_data,

                agent_results=
                    agent_results,

                fusion_result=
                    fusion_result,

                conflicts=
                    conflicts,
            )
        )

        # =========================================================================
        # STEP 6
        # =========================================================================

        self._log(
            "\nSTEP 6/6 → Decision Intelligence"
        )

        if self.decision_engine is not None:

            try:

                decision = (
                    self.decision_engine.decide(
                        agent_results=
                            agent_results,

                        conflicts=
                            conflicts,

                        fused_results=
                            fusion_result,

                        clinical_reasoning=
                            clinical_result,
                    )
                )

            except Exception as e:

                self._log(
                    f"Decision engine failed: {e}"
                )

                decision = (
                    self._fallback_decision(
                        agent_results,
                        conflicts,
                        clinical_result,
                    )
                )

        else:

            decision = (
                self._fallback_decision(
                    agent_results,
                    conflicts,
                    clinical_result,
                )
            )

        # =========================================================================
        # SERIALIZATION
        # =========================================================================

        serialized_agents = {}

        if AgentResult is not None:

            for result in agent_results:

                try:

                    serialized_agents[
                        result.agent_id
                    ] = result.to_dict()

                except Exception:

                    pass

        else:

            serialized_agents = raw_results

        serialized_agents[
            "ClinicalReasoningAgent"
        ] = clinical_result

        # =========================================================================
        # COMPLETED AGENTS
        # =========================================================================

        completed_agents = []

        for name, result in raw_results.items():

            if result.get(
                "status"
            ) in [
                "success",
                "completed",
            ]:

                completed_agents.append(
                    name
                )

        # Clinical reasoning is separate from specialized agents.

        clinical_completed = (
            clinical_result.get(
                "status"
            )
            in [
                "success",
                "completed",
            ]
        )

        # =========================================================================
        # FINAL RESULT
        # =========================================================================

        final_result = {

            "system":
                "LiverAI-MultiAgent",

            "patient_id":
                patient_id,

            "timestamp":
                datetime.now().isoformat(),

            "status":
                "completed",

            "total_specialized_agents":
                5,

            "total_agents":
                6,

            "agents_completed":
                len(
                    completed_agents
                ),

            "clinical_reasoning_completed":
                clinical_completed,

            "completed_agent_names":
                completed_agents,

            "agents":
                serialized_agents,

            "adaptive_fusion":
                fusion_result,

            "conflicts":
                conflicts,

            "clinical_reasoning":
                clinical_result,

            "decision":
                decision,

            "execution_log":
                self.execution_log,
        }

        self.last_results = final_result
        self.last_assessment = final_result

        self._log(
            "\n"
            + "=" * 80
        )

        self._log(
            "LIVERAI PIPELINE COMPLETED"
        )

        self._log(
            "=" * 80
        )

        return final_result

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_system_status(self):

        status = {}

        for name, agent in self.agents.items():

            status[name] = {

                "loaded":
                    agent is not None,

                "class":
                    (
                        agent.__class__.__name__
                        if agent is not None
                        else None
                    ),
            }

        return status

    # =========================================================================
    # RESULTS
    # =========================================================================

    def get_last_results(self):

        return self.last_results

    def get_last_assessment(self):

        return self.last_assessment

    def get_execution_log(self):

        return self.execution_log

    # =========================================================================
    # UTILITY
    # =========================================================================

    @staticmethod
    def _clip(value):

        try:

            value = float(value)

        except (
            TypeError,
            ValueError,
        ):

            value = 0.0

        if not np.isfinite(value):

            value = 0.0

        return max(
            0.0,
            min(
                1.0,
                value
            )
        )
