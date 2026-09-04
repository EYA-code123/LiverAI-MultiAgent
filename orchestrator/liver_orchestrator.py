```python
# =============================================================================
# LiverAI-MultiAgent
# COMPLETE MULTI-MODAL MULTI-AGENT ORCHESTRATOR
# =============================================================================

import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional


# =============================================================================
# STANDARD SCHEMA
# =============================================================================

try:
    from orchestrator.schemas import AgentResult
except Exception:
    AgentResult = None


# =============================================================================
# COORDINATION MODULES
# =============================================================================

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
# ORCHESTRATOR
# =============================================================================

class LiverAIOrchestrator:
    """
    Central orchestrator for the LiverAI multi-agent system.

    Six agents:

        1. Fatty Liver Classification
        2. Fibrosis Prediction
        3. Cirrhosis Classification
        4. Tumor Classification
        5. Liver Segmentation
        6. Clinical Reasoning

    Architecture:

        Patient Input
              |
              v
        Specialized Agents
              |
              v
        Standardization
              |
              v
        Trust Computation
              |
              v
        Adaptive Fusion
              |
              v
        Conflict Detection
              |
              v
        Clinical Reasoning
              |
              v
        Decision Engine
              |
              v
        Unified LiverAI Report

    Important:
        Each agent receives ONLY its own modality/features.
    """

    # =========================================================================
    # MODEL PATHS
    # =========================================================================

    DEFAULT_MODEL_PATHS = {

        "fatty_liver":
            "/content/drive/MyDrive/"
            "Fatty_Liver_Dataset/models/"
            "FattyLiver_LightGBM.pkl",

        "fibrosis":
            "/content/drive/MyDrive/"
            "Fibrosis Agent/XGBoost_model/"
            "xgboost_nafld.pkl",

        "cirrhosis":
            "/content/drive/MyDrive/"
            ".Cirrhosis Agent/XGBoost_model/"
            "XGBoost_Cirrhosis_fixed.joblib",

        "tumor_classification":
            "/content/drive/MyDrive/"
            "models/tumor/"
            "efficientnet_b0_best.pth",

        "liver_segmentation":
            "/content/drive/MyDrive/"
            "Liver Segmentation Agent/models/"
            "SegResNet3D_Liver_best.pth",

        "clinical_reasoning":
            "/content/drive/MyDrive/"
            "Clinical Reasoning Agent/"
            "tabtransformer_bupa",
    }

    # =========================================================================
    # REQUIRED FEATURES
    # =========================================================================

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

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        model_paths: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
    ):

        self.name = "LiverAI Multi-Agent Orchestrator"

        # ---------------------------------------------------------------------
        # PATHS
        # ---------------------------------------------------------------------

        self.model_paths = self.DEFAULT_MODEL_PATHS.copy()

        if model_paths is not None:
            self.model_paths.update(model_paths)

        self.device = device

        # ---------------------------------------------------------------------
        # AGENTS
        # ---------------------------------------------------------------------

        self.fatty_liver_agent = None
        self.fibrosis_agent = None
        self.cirrhosis_agent = None
        self.tumor_agent = None
        self.segmentation_agent = None
        self.clinical_agent = None

        # ---------------------------------------------------------------------
        # COORDINATION
        # ---------------------------------------------------------------------

        self.trust_manager = None
        self.adaptive_fusion = None
        self.conflict_detector = None
        self.decision_engine = None

        # ---------------------------------------------------------------------
        # STATE
        # ---------------------------------------------------------------------

        self.last_results = {}
        self.last_assessment = None
        self.execution_log = []

        # ---------------------------------------------------------------------
        # LOAD AGENTS
        # ---------------------------------------------------------------------

        self._load_all_agents()

        # ---------------------------------------------------------------------
        # REGISTRY
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
        # LOAD COORDINATION MODULES
        # ---------------------------------------------------------------------

        self._load_coordination_modules()

        # ---------------------------------------------------------------------
        # DISPLAY STATUS
        # ---------------------------------------------------------------------

        self._print_system_status()

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
    # AGENT LOADING
    # =========================================================================

    def _load_all_agents(self):

        print("\n" + "=" * 80)
        print("LOADING LIVERAI AGENTS")
        print("=" * 80)

        self._load_fatty_liver_agent()
        self._load_fibrosis_agent()
        self._load_cirrhosis_agent()
        self._load_tumor_agent()
        self._load_segmentation_agent()
        self._load_clinical_reasoning_agent()

    # -------------------------------------------------------------------------

    def _load_fatty_liver_agent(self):

        try:

            import joblib

            from agents.fatty_liver_agent import (
                FattyLiverAgent
            )

            model = joblib.load(
                self.model_paths["fatty_liver"]
            )

            self.fatty_liver_agent = FattyLiverAgent(
                model
            )

            print("✓ Fatty Liver Agent loaded")

        except Exception as e:

            print(
                "✗ Fatty Liver Agent failed: "
                f"{type(e).__name__}: {e}"
            )

            self.fatty_liver_agent = None

    # -------------------------------------------------------------------------

    def _load_fibrosis_agent(self):

        try:

            import joblib

            from agents.fibrosis_agent import (
                FibrosisAgent
            )

            model = joblib.load(
                self.model_paths["fibrosis"]
            )

            self.fibrosis_agent = FibrosisAgent(
                model
            )

            print("✓ Fibrosis Agent loaded")

        except Exception as e:

            print(
                "✗ Fibrosis Agent failed: "
                f"{type(e).__name__}: {e}"
            )

            self.fibrosis_agent = None

    # -------------------------------------------------------------------------

    def _load_cirrhosis_agent(self):

        try:

            from agents.cirrhosis_agent import (
                CirrhosisAgent
            )

            self.cirrhosis_agent = CirrhosisAgent(
                self.model_paths["cirrhosis"]
            )

            print("✓ Cirrhosis Agent loaded")

        except Exception as e:

            print(
                "✗ Cirrhosis Agent failed: "
                f"{type(e).__name__}: {e}"
            )

            self.cirrhosis_agent = None

    # -------------------------------------------------------------------------

    def _load_tumor_agent(self):

        try:

            from agents.tumor_classification_agent import (
                TumorClassificationAgent
            )

            self.tumor_agent = TumorClassificationAgent(
                self.model_paths["tumor_classification"]
            )

            print(
                "✓ Tumor Classification Agent loaded"
            )

        except Exception as e:

            print(
                "✗ Tumor Classification Agent failed: "
                f"{type(e).__name__}: {e}"
            )

            self.tumor_agent = None

    # -------------------------------------------------------------------------

    def _load_segmentation_agent(self):

        try:

            from agents.liver_segmentation_agent import (
                LiverSegmentationAgent
            )

            kwargs = {
                "model_path":
                    self.model_paths["liver_segmentation"]
            }

            if self.device is not None:
                kwargs["device"] = self.device

            self.segmentation_agent = (
                LiverSegmentationAgent(**kwargs)
            )

            print(
                "✓ Liver Segmentation Agent loaded"
            )

        except Exception as e:

            print(
                "✗ Liver Segmentation Agent failed: "
                f"{type(e).__name__}: {e}"
            )

            self.segmentation_agent = None

    # -------------------------------------------------------------------------

    def _load_clinical_reasoning_agent(self):

        try:

            from agents.clinical_reasoning_agent import (
                ClinicalReasoningAgent
            )

            self.clinical_agent = ClinicalReasoningAgent(
                self.model_paths["clinical_reasoning"]
            )

            print(
                "✓ Clinical Reasoning Agent loaded"
            )

        except Exception as e:

            print(
                "✗ Clinical Reasoning Agent failed: "
                f"{type(e).__name__}: {e}"
            )

            self.clinical_agent = None

    # =========================================================================
    # COORDINATION MODULE LOADING
    # =========================================================================

    def _load_coordination_modules(self):

        try:

            if TrustManager is not None:
                self.trust_manager = TrustManager()

        except Exception as e:

            self._log(
                f"TrustManager initialization failed: {e}"
            )

        try:

            if AdaptiveFusion is not None:
                self.adaptive_fusion = AdaptiveFusion()

        except Exception as e:

            self._log(
                f"AdaptiveFusion initialization failed: {e}"
            )

        try:

            if ConflictDetector is not None:
                self.conflict_detector = ConflictDetector()

        except Exception as e:

            self._log(
                f"ConflictDetector initialization failed: {e}"
            )

        try:

            if DecisionEngine is not None:
                self.decision_engine = DecisionEngine()

        except Exception as e:

            self._log(
                f"DecisionEngine initialization failed: {e}"
            )

    # =========================================================================
    # SYSTEM STATUS
    # =========================================================================

    def _print_system_status(self):

        print("\n" + "=" * 80)
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
                f"  {name:<25}: {status}"
            )

        print("\nCoordination Modules:")

        print(
            f"  Trust Manager           : "
            f"{'READY' if self.trust_manager else 'FALLBACK'}"
        )

        print(
            f"  Adaptive Fusion         : "
            f"{'READY' if self.adaptive_fusion else 'FALLBACK'}"
        )

        print(
            f"  Conflict Detector       : "
            f"{'READY' if self.conflict_detector else 'FALLBACK'}"
        )

        print(
            f"  Decision Engine         : "
            f"{'READY' if self.decision_engine else 'FALLBACK'}"
        )

        print("=" * 80)

    # =========================================================================
    # INPUT HELPERS
    # =========================================================================

    @staticmethod
    def _has_features(
        data,
        required_features,
    ):

        if not isinstance(data, dict):
            return False

        return all(
            feature in data
            for feature in required_features
        )

    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_modality(
        patient_data,
        key,
        required_features=None,
    ):
        """
        Resolve nested or legacy flat input.

        Preferred:

            patient_data = {
                "fatty_liver": {...},
                "fibrosis": {...},
                "cirrhosis": {...},
                "tumor": image,
                "segmentation": volume,
                "clinical_reasoning": {...}
            }

        Legacy flat dictionaries are also supported.
        """

        if patient_data is None:
            return None

        if not isinstance(patient_data, dict):
            return patient_data

        # Preferred nested structure
        if key in patient_data:
            return patient_data[key]

        # Legacy flat structure
        if required_features is not None:

            if any(
                feature in patient_data
                for feature in required_features
            ):
                return {
                    feature: patient_data[feature]
                    for feature in required_features
                    if feature in patient_data
                }

        return None

    # =========================================================================
    # RESULT HELPERS
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

        return max(
            0.0,
            min(
                1.0,
                value
            )
        )

    # -------------------------------------------------------------------------

    def _empty_result(
        self,
        agent_id,
        task_type,
        status="not_run",
        error=None,
        reason=None,
    ):

        return {

            "agent_id":
                agent_id,

            "agent":
                agent_id,

            "task_type":
                task_type,

            "status":
                status,

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

            "reason":
                reason,
        }

    # =========================================================================
    # GENERIC AGENT EXECUTION
    # =========================================================================

    def _execute_agent(
        self,
        agent_id,
        agent,
        input_data,
        task_type,
    ):

        if agent is None:

            self._log(
                f"[{agent_id}] NOT AVAILABLE"
            )

            return self._empty_result(
                agent_id,
                task_type,
                status="not_available",
                error="Agent not loaded.",
            )

        if input_data is None:

            self._log(
                f"[{agent_id}] NO INPUT -> SKIPPED"
            )

            return self._empty_result(
                agent_id,
                task_type,
                status="not_run",
                reason="Required input not provided.",
            )

        self._log(
            f"\n[{agent_id}] START"
        )

        start_time = time.time()

        try:

            # ---------------------------------------------------------------
            # RUN / PREDICT / ANALYZE
            # ---------------------------------------------------------------

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
                    f"{agent_id} has no supported "
                    "predict(), run(), or analyze() method."
                )

            # ---------------------------------------------------------------
            # NORMALIZE
            # ---------------------------------------------------------------

            if result is None:
                result = {}

            if not isinstance(result, dict):

                result = {
                    "prediction": result
                }

            result = dict(result)

            elapsed_ms = (
                time.time() - start_time
            ) * 1000.0

            result.setdefault(
                "agent_id",
                agent_id
            )

            result.setdefault(
                "agent",
                agent_id
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

            # ---------------------------------------------------------------
            # NORMALIZE NUMERICAL FIELDS
            # ---------------------------------------------------------------

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
                    1.0
                )
            )

            result["missing_data_ratio"] = self._clip(
                result.get(
                    "missing_data_ratio",
                    0.0
                )
            )

            try:

                result["latency_ms"] = float(
                    result.get(
                        "latency_ms",
                        elapsed_ms
                    )
                )

            except Exception:

                result["latency_ms"] = elapsed_ms

            self._log(
                f"[{agent_id}] COMPLETED "
                f"| status={result['status']} "
                f"| confidence="
                f"{result['confidence']:.3f}"
            )

            return result

        except Exception as e:

            elapsed_ms = (
                time.time() - start_time
            ) * 1000.0

            self._log(
                f"[{agent_id}] ERROR: {e}"
            )

            traceback.print_exc()

            return self._empty_result(
                agent_id,
                task_type,
                status="error",
                error=str(e),
            )

    # =========================================================================
    # SPECIALIZED AGENTS
    # =========================================================================

    def run_fatty_liver(
        self,
        clinical_data,
    ):

        if not self._has_features(
            clinical_data,
            self.FATTY_FEATURES,
        ):

            return self._empty_result(
                "FattyLiverAgent",
                "fatty_liver_classification",
                status="not_run",
                reason="Missing fatty liver features.",
            )

        data = {
            feature: clinical_data[feature]
            for feature in self.FATTY_FEATURES
        }

        return self._execute_agent(
            "FattyLiverAgent",
            self.fatty_liver_agent,
            data,
            "fatty_liver_classification",
        )

    # -------------------------------------------------------------------------

    def run_fibrosis(
        self,
        fibrosis_input,
    ):

        if not self._has_features(
            fibrosis_input,
            self.FIBROSIS_FEATURES,
        ):

            return self._empty_result(
                "FibrosisAgent",
                "fibrosis_prediction",
                status="not_run",
                reason="Missing fibrosis features.",
            )

        data = {
            feature: fibrosis_input[feature]
            for feature in self.FIBROSIS_FEATURES
        }

        return self._execute_agent(
            "FibrosisAgent",
            self.fibrosis_agent,
            data,
            "fibrosis_prediction",
        )

    # -------------------------------------------------------------------------

    def run_cirrhosis(
        self,
        cirrhosis_input,
    ):

        if not self._has_features(
            cirrhosis_input,
            self.CIRRHOSIS_FEATURES,
        ):

            return self._empty_result(
                "CirrhosisAgent",
                "cirrhosis_classification",
                status="not_run",
                reason="Missing cirrhosis features.",
            )

        data = {
            feature: cirrhosis_input[feature]
            for feature in self.CIRRHOSIS_FEATURES
        }

        return self._execute_agent(
            "CirrhosisAgent",
            self.cirrhosis_agent,
            data,
            "cirrhosis_classification",
        )

    # -------------------------------------------------------------------------

    def run_tumor_classification(
        self,
        image,
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
        volume,
    ):

        return self._execute_agent(
            "LiverSegmentationAgent",
            self.segmentation_agent,
            volume,
            "liver_segmentation",
        )

    # =========================================================================
    # SPECIALIZED PIPELINE
    # =========================================================================

    def run_specialized_agents(
        self,
        patient_data=None,
        clinical_data=None,
        fibrosis_input=None,
        cirrhosis_input=None,
        image=None,
        volume=None,
    ):

        self._log(
            "\n" + "=" * 80
        )

        self._log(
            "STEP 1/6 -> SPECIALIZED AGENTS"
        )

        self._log(
            "=" * 80
        )

        # ---------------------------------------------------------------------
        # RESOLVE INPUTS
        # ---------------------------------------------------------------------

        if patient_data is not None:

            fatty_input = self._extract_modality(
                patient_data,
                "fatty_liver",
                self.FATTY_FEATURES,
            )

            fibrosis_data = self._extract_modality(
                patient_data,
                "fibrosis",
                self.FIBROSIS_FEATURES,
            )

            cirrhosis_data = self._extract_modality(
                patient_data,
                "cirrhosis",
                self.CIRRHOSIS_FEATURES,
            )

            tumor_data = self._extract_modality(
                patient_data,
                "tumor",
            )

            segmentation_data = self._extract_modality(
                patient_data,
                "segmentation",
            )

        else:

            fatty_input = clinical_data
            fibrosis_data = fibrosis_input
            cirrhosis_data = cirrhosis_input
            tumor_data = image
            segmentation_data = volume

        # ---------------------------------------------------------------------
        # FAT
        # ---------------------------------------------------------------------

        self._log(
            "\n--- 1/5 FATty Liver ---"
        )

        fatty_result = self.run_fatty_liver(
            fatty_input
        )

        # ---------------------------------------------------------------------
        # FIBROSIS
        # ---------------------------------------------------------------------

        self._log(
            "\n--- 2/5 Fibrosis ---"
        )

        fibrosis_result = self.run_fibrosis(
            fibrosis_data
        )

        # ---------------------------------------------------------------------
        # CIRRHOSIS
        # ---------------------------------------------------------------------

        self._log(
            "\n--- 3/5 Cirrhosis ---"
        )

        cirrhosis_result = self.run_cirrhosis(
            cirrhosis_data
        )

        # ---------------------------------------------------------------------
        # TUMOR
        # ---------------------------------------------------------------------

        self._log(
            "\n--- 4/5 Tumor Classification ---"
        )

        tumor_result = self.run_tumor_classification(
            tumor_data
        )

        # ---------------------------------------------------------------------
        # SEGMENTATION
        # ---------------------------------------------------------------------

        self._log(
            "\n--- 5/5 Liver Segmentation ---"
        )

        segmentation_result = self.run_liver_segmentation(
            segmentation_data
        )

        return {

            "fatty_liver":
                fatty_result,

            "fibrosis":
                fibrosis_result,

            "cirrhosis":
                cirrhosis_result,

            "tumor_classification":
                tumor_result,

            "liver_segmentation":
                segmentation_result,
        }

    # =========================================================================
    # STANDARDIZATION
    # =========================================================================

    def _to_agent_results(
        self,
        raw_results,
    ):

        if AgentResult is None:

            self._log(
                "AgentResult unavailable -> "
                "using dictionaries."
            )

            return []

        agent_results = []

        for result in raw_results.values():

            try:

                standardized = AgentResult.from_dict(
                    result
                )

                # -------------------------------------------------------------
                # TRUST
                # -------------------------------------------------------------

                if self.trust_manager is not None:

                    try:

                        trust = (
                            self.trust_manager.compute_trust(
                                agent_id=
                                    standardized.agent_id,

                                confidence=
                                    standardized.confidence,

                                quality=
                                    standardized.quality,

                                uncertainty=
                                    standardized.uncertainty,

                                missing_data_ratio=
                                    standardized.missing_data_ratio,

                                modality_available=(
                                    standardized.status
                                    in (
                                        "success",
                                        "completed",
                                    )
                                ),
                            )
                        )

                        standardized.trust = (
                            self._clip(trust)
                        )

                    except Exception as e:

                        self._log(
                            "Trust computation failed for "
                            f"{standardized.agent_id}: {e}"
                        )

                        standardized.trust = (
                            standardized.confidence
                        )

                else:

                    standardized.trust = (
                        standardized.confidence
                    )

                agent_results.append(
                    standardized
                )

            except Exception as e:

                self._log(
                    f"AgentResult conversion failed: {e}"
                )

        return agent_results

    # =========================================================================
    # CONVERT AgentResult -> DICT
    # =========================================================================

    @staticmethod
    def _agent_results_to_dicts(
        agent_results,
    ):

        output = []

        for result in agent_results:

            if hasattr(
                result,
                "to_dict"
            ):

                output.append(
                    result.to_dict()
                )

            elif isinstance(
                result,
                dict
            ):

                output.append(
                    dict(result)
                )

        return output

    # =========================================================================
    # FALLBACK FUSION
    # =========================================================================

    def _fallback_fusion(
        self,
        agent_results,
    ):

        evidence = []

        for result in agent_results:

            if isinstance(
                result,
                dict
            ):

                status = result.get(
                    "status",
                    "success"
                )

                prediction = result.get(
                    "prediction"
                )

                agent_id = result.get(
                    "agent_id",
                    result.get(
                        "agent",
                        "unknown"
                    )
                )

                confidence = self._clip(
                    result.get(
                        "confidence",
                        0.0
                    )
                )

                trust = result.get(
                    "trust"
                )

            else:

                status = getattr(
                    result,
                    "status",
                    "success"
                )

                prediction = getattr(
                    result,
                    "prediction",
                    None
                )

                agent_id = getattr(
                    result,
                    "agent_id",
                    "unknown"
                )

                confidence = self._clip(
                    getattr(
                        result,
                        "confidence",
                        0.0
                    )
                )

                trust = getattr(
                    result,
                    "trust",
                    None
                )

            if status not in (
                "success",
                "completed",
            ):

                continue

            if prediction is None:

                continue

            if trust is None:

                trust = confidence

            evidence.append(
                {
                    "agent_id":
                        agent_id,

                    "prediction":
                        prediction,

                    "confidence":
                        confidence,

                    "trust":
                        self._clip(trust),
                }
            )

        return {

            "status":
                "success",

            "method":
                "fallback_weighted_evidence",

            "evidence":
                evidence,

            "num_valid_agents":
                len(evidence),

            "weights":
                {
                    item["agent_id"]:
                        item["trust"]
                    for item in evidence
                },
        }

    # =========================================================================
    # FALLBACK CONFLICTS
    # =========================================================================

    def _fallback_conflicts(
        self,
        agent_results,
    ):

        conflicts = []

        groups = {}

        for result in agent_results:

            groups.setdefault(
                result.task_type,
                []
            ).append(result)

        for task_type, group in groups.items():

            predictions = [
                str(item.prediction)
                for item in group
                if item.prediction is not None
            ]

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
                                item.agent_id
                                for item in group
                            ],
                    }
                )

        return conflicts

    # =========================================================================
    # FALLBACK DECISION
    # =========================================================================

    def _fallback_decision(
        self,
        agent_results,
        conflicts,
        clinical_reasoning,
    ):

        valid = [
            result
            for result in agent_results
            if result.status in (
                "success",
                "completed",
            )
            and result.prediction is not None
        ]

        if not valid:

            return {

                "status":
                    "insufficient_evidence",

                "decision_level":
                    "UNCERTAIN",

                "prediction":
                    None,

                "confidence":
                    0.0,

                "risk_score":
                    1.0,

                "request_additional_tests":
                    True,

                "num_valid_agents":
                    0,

                "note":
                    (
                        "Insufficient valid model outputs. "
                        "Research/engineering output only."
                    ),
            }

        mean_confidence = sum(
            self._clip(
                result.confidence
            )
            for result in valid
        ) / len(valid)

        mean_trust = sum(
            self._clip(
                result.trust
                if result.trust is not None
                else result.confidence
            )
            for result in valid
        ) / len(valid)

        mean_quality = sum(
            self._clip(
                result.quality
            )
            for result in valid
        ) / len(valid)

        if mean_confidence >= 0.80 and mean_trust >= 0.70:

            decision_level = "HIGH"

        elif mean_confidence >= 0.55:

            decision_level = "MODERATE"

        else:

            decision_level = "UNCERTAIN"

        if decision_level == "UNCERTAIN":

            request_additional_tests = True

        else:

            request_additional_tests = (
                len(conflicts) > 0
            )

        prediction = None

        if isinstance(
            clinical_reasoning,
            dict
        ):

            prediction = clinical_reasoning.get(
                "prediction"
            )

        if prediction is None:

            best = max(
                valid,
                key=lambda x:
                    self._clip(
                        x.trust
                        if x.trust is not None
                        else 0.0
                    )
                    *
                    self._clip(
                        x.confidence
                    )
            )

            prediction = best.prediction

        risk_score = (
            0.40 * (1.0 - mean_confidence)
            +
            0.30 * (1.0 - mean_trust)
            +
            0.20 * min(
                1.0,
                len(conflicts) / max(
                    1,
                    len(valid)
                )
            )
            +
            0.10 * (1.0 - mean_quality)
        )

        risk_score = self._clip(
            risk_score
        )

        return {

            "status":
                "completed",

            "prediction":
                prediction,

            "decision_level":
                decision_level,

            "confidence":
                float(mean_confidence),

            "uncertainty":
                float(1.0 - mean_confidence),

            "trust":
                float(mean_trust),

            "quality":
                float(mean_quality),

            "risk_score":
                float(risk_score),

            "request_additional_tests":
                request_additional_tests,

            "num_valid_agents":
                len(valid),

            "num_conflicts":
                len(conflicts),

            "note":
                (
                    "System-level evidence aggregation only. "
                    "Not a medical diagnosis."
                ),
        }

    # =========================================================================
    # CLINICAL REASONING
    # =========================================================================

    def run_clinical_reasoning(
        self,
        clinical_input,
    ):

        self._log(
            "\n--- 5/6 Clinical Reasoning ---"
        )

        if self.clinical_agent is None:

            return self._empty_result(
                "ClinicalReasoningAgent",
                "clinical_reasoning",
                status="not_available",
                error="Clinical reasoning agent not loaded.",
            )

        if not self._has_features(
            clinical_input,
            self.CLINICAL_FEATURES,
        ):

            return self._empty_result(
                "ClinicalReasoningAgent",
                "clinical_reasoning",
                status="not_run",
                reason="Missing clinical reasoning features.",
            )

        data = {
            feature: clinical_input[feature]
            for feature in self.CLINICAL_FEATURES
        }

        return self._execute_agent(
            "ClinicalReasoningAgent",
            self.clinical_agent,
            data,
            "clinical_reasoning",
        )

    # =========================================================================
    # COMPLETE PIPELINE
    # =========================================================================

    def run(
        self,
        patient_id,
        patient_data=None,
        clinical_data=None,
        fibrosis_input=None,
        cirrhosis_input=None,
        image=None,
        volume=None,
        clinical_reasoning_input=None,
    ):
        """
        Main entry point.

        Recommended:

            result = orchestrator.run(
                patient_id="PATIENT_001",
                patient_data={
                    "fatty_liver": {...},
                    "fibrosis": {...},
                    "cirrhosis": {...},
                    "tumor": image,
                    "segmentation": volume,
                    "clinical_reasoning": {...}
                }
            )

        Legacy parameters remain supported.
        """

        self.execution_log = []

        start_time = time.time()

        self._log(
            "\n" + "=" * 80
        )

        self._log(
            f"LIVERAI ANALYSIS - PATIENT {patient_id}"
        )

        self._log(
            "=" * 80
        )

        # =====================================================================
        # STEP 1
        # =====================================================================

        raw_specialized = (
            self.run_specialized_agents(
                patient_data=patient_data,
                clinical_data=clinical_data,
                fibrosis_input=fibrosis_input,
                cirrhosis_input=cirrhosis_input,
                image=image,
                volume=volume,
            )
        )

        # =====================================================================
        # STEP 2 - STANDARDIZATION + TRUST
        # =====================================================================

        self._log(
            "\n" + "=" * 80
        )

        self._log(
            "STEP 2/6 -> STANDARDIZATION + TRUST"
        )

        self._log(
            "=" * 80
        )

        agent_results = self._to_agent_results(
            raw_specialized
        )

        # Convert to dictionaries for modules
        agent_dicts = (
            self._agent_results_to_dicts(
                agent_results
            )
        )

        # =====================================================================
        # STEP 3 - ADAPTIVE FUSION
        # =====================================================================

        self._log(
            "\n" + "=" * 80
        )

        self._log(
            "STEP 3/6 -> ADAPTIVE EVIDENCE FUSION"
        )

        self._log(
            "=" * 80
        )

        if self.adaptive_fusion is not None:

            try:

                # IMPORTANT:
                # AdaptiveFusion expects dictionaries.
                fusion_result = (
                    self.adaptive_fusion.fuse(
                        agent_dicts
                    )
                )

            except Exception as e:

                self._log(
                    f"Adaptive fusion failed: {e}"
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
        # STEP 4 - CONFLICT DETECTION
        # =====================================================================

        self._log(
            "\n" + "=" * 80
        )

        self._log(
            "STEP 4/6 -> CONFLICT DETECTION"
        )

        self._log(
            "=" * 80
        )

        if self.conflict_detector is not None:

            try:

                # IMPORTANT:
                # ConflictDetector expects AgentResult objects.
                conflicts = (
                    self.conflict_detector.detect(
                        agent_results
                    )
                )

                if conflicts is None:
                    conflicts = []

            except Exception as e:

                self._log(
                    f"Conflict detection failed: {e}"
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

        # =====================================================================
        # STEP 5 - CLINICAL REASONING
        # =====================================================================

        self._log(
            "\n" + "=" * 80
        )

        self._log(
            "STEP 5/6 -> CLINICAL REASONING"
        )

        self._log(
            "=" * 80
        )

        # Resolve clinical input

        if clinical_reasoning_input is not None:

            clinical_input = (
                clinical_reasoning_input
            )

        elif patient_data is not None:

            clinical_input = (
                self._extract_modality(
                    patient_data,
                    "clinical_reasoning",
                    self.CLINICAL_FEATURES,
                )
            )

        else:

            clinical_input = clinical_data

        clinical_result = (
            self.run_clinical_reasoning(
                clinical_input
            )
        )

        # =====================================================================
        # STEP 6 - DECISION ENGINE
        # =====================================================================

        self._log(
            "\n" + "=" * 80
        )

        self._log(
            "STEP 6/6 -> DECISION ENGINE"
        )

        self._log(
            "=" * 80
        )

        # Add clinical reasoning to dictionary results
        all_results = list(
            agent_dicts
        )

        clinical_dict = dict(
            clinical_result
        )

        if clinical_dict.get(
            "status"
        ) in (
            "success",
            "completed",
        ):

            all_results.append(
                clinical_dict
            )

        if self.decision_engine is not None:

            try:

                # IMPORTANT:
                # Current DecisionEngine API:
                #
                # decide(
                #     results,
                #     conflicts=None,
                #     reasoning=None
                # )
                decision = (
                    self.decision_engine.decide(
                        all_results,
                        conflicts,
                        clinical_dict,
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
                        clinical_dict,
                    )
                )

        else:

            decision = (
                self._fallback_decision(
                    agent_results,
                    conflicts,
                    clinical_dict,
                )
            )

        # =====================================================================
        # SERIALIZE AGENTS
        # =====================================================================

        serialized_agents = {}

        for result in agent_dicts:

            agent_id = result.get(
                "agent_id",
                result.get(
                    "agent",
                    "unknown"
                )
            )

            serialized_agents[
                agent_id
            ] = result

        serialized_agents[
            "ClinicalReasoningAgent"
        ] = clinical_dict

        # =====================================================================
        # COUNTS
        # =====================================================================

        completed_names = []

        for name, result in serialized_agents.items():

            if result.get(
                "status"
            ) in (
                "success",
                "completed",
            ):

                completed_names.append(
                    name
                )

        total_inference_time = (
            time.time() - start_time
        )

        # =====================================================================
        # FINAL STATUS
        # =====================================================================

        successful_count = len(
            completed_names
        )

        if successful_count == 6:

            final_status = "completed"

        elif successful_count > 0:

            final_status = "partial"

        else:

            final_status = "failed"

        # =====================================================================
        # FINAL RESULT
        # =====================================================================

        final_result = {

            "system":
                "LiverAI-MultiAgent",

            "patient_id":
                patient_id,

            "timestamp":
                datetime.now().isoformat(),

            "status":
                final_status,

            # -----------------------------------------------------------------
            # AGENT COUNTS
            # -----------------------------------------------------------------

            "total_specialized_agents":
                5,

            "total_agents":
                6,

            "agents_completed":
                successful_count,

            "completed_agent_names":
                completed_names,

            # -----------------------------------------------------------------
            # AGENTS
            # -----------------------------------------------------------------

            "agents":
                serialized_agents,

            # -----------------------------------------------------------------
            # COORDINATION
            # -----------------------------------------------------------------

            "adaptive_fusion":
                fusion_result,

            "conflicts":
                conflicts,

            "clinical_reasoning":
                clinical_dict,

            "decision":
                decision,

            # -----------------------------------------------------------------
            # INPUT AVAILABILITY
            # -----------------------------------------------------------------

            "input_availability":
                {
                    "patient_data":
                        patient_data is not None,

                    "fatty_liver":
                        (
                            self._extract_modality(
                                patient_data,
                                "fatty_liver",
                                self.FATTY_FEATURES,
                            )
                            is not None
                            if patient_data is not None
                            else clinical_data is not None
                        ),

                    "fibrosis":
                        (
                            self._extract_modality(
                                patient_data,
                                "fibrosis",
                                self.FIBROSIS_FEATURES,
                            )
                            is not None
                            if patient_data is not None
                            else fibrosis_input is not None
                        ),

                    "cirrhosis":
                        (
                            self._extract_modality(
                                patient_data,
                                "cirrhosis",
                                self.CIRRHOSIS_FEATURES,
                            )
                            is not None
                            if patient_data is not None
                            else cirrhosis_input is not None
                        ),

                    "tumor":
                        (
                            self._extract_modality(
                                patient_data,
                                "tumor",
                            )
                            is not None
                            if patient_data is not None
                            else image is not None
                        ),

                    "segmentation":
                        (
                            self._extract_modality(
                                patient_data,
                                "segmentation",
                            )
                            is not None
                            if patient_data is not None
                            else volume is not None
                        ),

                    "clinical_reasoning":
                        clinical_input is not None,
                },

            # -----------------------------------------------------------------
            # EXECUTION
            # -----------------------------------------------------------------

            "execution":
                {
                    "total_inference_time_seconds":
                        total_inference_time,

                    "execution_log":
                        self.execution_log,
                },

            # -----------------------------------------------------------------
            # SYSTEM NOTE
            # -----------------------------------------------------------------

            "note":
                (
                    "LiverAI performs model-based evidence aggregation "
                    "for research and engineering purposes. "
                    "Outputs are not medical diagnoses."
                ),
        }

        # =====================================================================
        # SAVE STATE
        # =====================================================================

        self.last_results = final_result

        self.last_assessment = final_result

        # =====================================================================
        # FINAL LOG
        # =====================================================================

        self._log(
            "\n" + "=" * 80
        )

        self._log(
            "LIVERAI PIPELINE COMPLETED"
        )

        self._log(
            f"Agents completed: "
            f"{successful_count}/6"
        )

        self._log(
            f"Final status: "
            f"{final_status}"
        )

        self._log(
            "=" * 80
        )

        return final_result

    # =========================================================================
    # SYSTEM STATUS
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

        status[
            "coordination"
        ] = {

            "trust_manager":
                self.trust_manager is not None,

            "adaptive_fusion":
                self.adaptive_fusion is not None,

            "conflict_detector":
                self.conflict_detector is not None,

            "decision_engine":
                self.decision_engine is not None,
        }

        return status

    # =========================================================================
    # LAST RESULTS
    # =========================================================================

    def get_last_results(self):

        return self.last_results

    # =========================================================================

    def get_last_assessment(self):

        return self.last_assessment

    # =========================================================================

    def get_execution_log(self):

        return self.execution_log


# =============================================================================
# DIRECT TEST
# =============================================================================

if __name__ == "__main__":

    orchestrator = LiverAIOrchestrator()

    print(
        orchestrator.get_system_status()
    )
```
