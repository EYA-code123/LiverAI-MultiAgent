# =============================================================================
# LiverAI-MultiAgent
# COMPLETE MULTI-AGENT ORCHESTRATOR
# =============================================================================

import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from orchestrator.schemas import AgentResult


# =============================================================================
# OPTIONAL COORDINATION MODULES
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


class LiverAIOrchestrator:
    """
    Main orchestration layer for the LiverAI multi-agent system.

    Specialized agents
    -------------------
    1. Fatty Liver
    2. Fibrosis
    3. Cirrhosis
    4. Tumor Classification
    5. Liver Segmentation

    Downstream reasoning
    --------------------
    6. Clinical Reasoning

    The orchestrator:
        - receives a unified patient_data dictionary
        - routes each modality to the correct agent
        - normalizes agent outputs
        - computes trust
        - performs adaptive fusion
        - detects conflicts
        - performs clinical reasoning
        - produces a final decision
    """

    # =========================================================================
    # MODEL PATHS
    # =========================================================================

    DEFAULT_MODEL_PATHS = {

        "fatty_liver":
            "/content/drive/MyDrive/Fatty_Liver_Dataset/models/"
            "FattyLiver_LightGBM.pkl",

        "fibrosis":
            "/content/drive/MyDrive/Fibrosis Agent/XGBoost_model/"
            "xgboost_nafld.pkl",

        "cirrhosis":
            "/content/drive/MyDrive/.Cirrhosis Agent/XGBoost_model/"
            "XGBoost_Cirrhosis_fixed.joblib",

        "tumor_classification":
            "/content/drive/MyDrive/models/tumor/"
            "efficientnet_b0_best.pth",

        "liver_segmentation":
            "/content/drive/MyDrive/Liver Segmentation Agent/models/"
            "SegResNet3D_Liver_best.pth",

        "clinical_reasoning":
            "/content/drive/MyDrive/Clinical Reasoning Agent/"
            "tabtransformer_bupa",
    }

    # =========================================================================
    # FEATURE DEFINITIONS
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

    def __init__(self, model_paths: Optional[Dict[str, str]] = None):

        self.name = "LiverAI Multi-Agent Orchestrator"

        paths = dict(self.DEFAULT_MODEL_PATHS)

        if model_paths:
            paths.update(model_paths)

        self.model_paths = paths

        # ---------------------------------------------------------------------
        # Agent placeholders
        # ---------------------------------------------------------------------

        self.fatty_liver_agent = None
        self.fibrosis_agent = None
        self.cirrhosis_agent = None
        self.tumor_agent = None
        self.segmentation_agent = None
        self.clinical_agent = None

        # ---------------------------------------------------------------------
        # Load agents
        # ---------------------------------------------------------------------

        self._load_fatty_liver_agent()
        self._load_fibrosis_agent()
        self._load_cirrhosis_agent()
        self._load_tumor_agent()
        self._load_segmentation_agent()
        self._load_clinical_agent()

        # ---------------------------------------------------------------------
        # Registry
        # ---------------------------------------------------------------------

        self.agents = {
            "fatty_liver": self.fatty_liver_agent,
            "fibrosis": self.fibrosis_agent,
            "cirrhosis": self.cirrhosis_agent,
            "tumor_classification": self.tumor_agent,
            "liver_segmentation": self.segmentation_agent,
            "clinical_reasoning": self.clinical_agent,
        }

        # ---------------------------------------------------------------------
        # Coordination modules
        # ---------------------------------------------------------------------

        self.trust_manager = (
            TrustManager() if TrustManager is not None else None
        )

        self.adaptive_fusion = (
            AdaptiveFusion() if AdaptiveFusion is not None else None
        )

        self.conflict_detector = (
            ConflictDetector() if ConflictDetector is not None else None
        )

        self.decision_engine = (
            DecisionEngine() if DecisionEngine is not None else None
        )

        print("=" * 70)
        print("LIVERAI ORCHESTRATOR INITIALIZED")
        print("=" * 70)

        loaded = [
            name
            for name, agent in self.agents.items()
            if agent is not None
        ]

        print(f"Loaded agents: {len(loaded)}/6")

        for name in loaded:
            print(f"  ✓ {name}")

        print("=" * 70)

    # =========================================================================
    # AGENT LOADERS
    # =========================================================================

    def _load_fatty_liver_agent(self):

        try:
            import joblib

            from agents.fatty_liver_agent import FattyLiverAgent

            model = joblib.load(
                self.model_paths["fatty_liver"]
            )

            self.fatty_liver_agent = FattyLiverAgent(model)

            print("✓ Fatty Liver Agent loaded")

        except Exception as e:

            print("✗ Fatty Liver Agent failed:")
            print(f"  {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------

    def _load_fibrosis_agent(self):

        try:
            import joblib

            from agents.fibrosis_agent import FibrosisAgent

            model = joblib.load(
                self.model_paths["fibrosis"]
            )

            self.fibrosis_agent = FibrosisAgent(model)

            print("✓ Fibrosis Agent loaded")

        except Exception as e:

            print("✗ Fibrosis Agent failed:")
            print(f"  {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------

    def _load_cirrhosis_agent(self):

        try:

            from agents.cirrhosis_agent import CirrhosisAgent

            self.cirrhosis_agent = CirrhosisAgent(
                self.model_paths["cirrhosis"]
            )

            print("✓ Cirrhosis Agent loaded")

        except Exception as e:

            print("✗ Cirrhosis Agent failed:")
            print(f"  {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------

    def _load_tumor_agent(self):

        try:

            from agents.tumor_classification_agent import (
                TumorClassificationAgent
            )

            self.tumor_agent = TumorClassificationAgent(
                self.model_paths["tumor_classification"]
            )

            print("✓ Tumor Classification Agent loaded")

        except Exception as e:

            print("✗ Tumor Classification Agent failed:")
            print(f"  {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------

    def _load_segmentation_agent(self):

        try:

            from agents.liver_segmentation_agent import (
                LiverSegmentationAgent
            )

            self.segmentation_agent = LiverSegmentationAgent(
                model_path=self.model_paths["liver_segmentation"]
            )

            print("✓ Liver Segmentation Agent loaded")

        except Exception as e:

            print("✗ Liver Segmentation Agent failed:")
            print(f"  {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------

    def _load_clinical_agent(self):

        try:

            from agents.clinical_reasoning_agent import (
                ClinicalReasoningAgent
            )

            self.clinical_agent = ClinicalReasoningAgent(
                self.model_paths["clinical_reasoning"]
            )

            print("✓ Clinical Reasoning Agent loaded")

        except Exception as e:

            print("✗ Clinical Reasoning Agent failed:")
            print(f"  {type(e).__name__}: {e}")

    # =========================================================================
    # INPUT EXTRACTION
    # =========================================================================

    @staticmethod
    def _extract_modality(
        patient_data: Optional[Dict[str, Any]],
        modality: str,
        legacy_value: Any = None,
    ):

        """
        Supports both:

        New format:
            patient_data = {
                "fatty_liver": {...},
                "fibrosis": {...},
                ...
            }

        Legacy format:
            individual arguments such as clinical_data,
            fibrosis_input, image and volume.
        """

        if isinstance(patient_data, dict):

            value = patient_data.get(modality)

            if value is not None:
                return value

        return legacy_value

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    @staticmethod
    def _has_value(value):

        if value is None:
            return False

        if isinstance(value, dict):
            return len(value) > 0

        return True

    # =========================================================================
    # AGENT EXECUTION
    # =========================================================================

    def _execute_agent(
        self,
        agent_name: str,
        agent,
        input_data: Any,
    ) -> Dict[str, Any]:

        start = datetime.now()

        if agent is None:

            return {
                "agent_id": agent_name,
                "task_type": agent_name,
                "status": "not_available",
                "prediction": None,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "missing_data_ratio": 1.0,
                "error": "Agent is not loaded.",
            }

        if not self._has_value(input_data):

            return {
                "agent_id": agent_name,
                "task_type": agent_name,
                "status": "not_run",
                "prediction": None,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "missing_data_ratio": 1.0,
                "error": "Required input modality was not provided.",
            }

        try:

            # -------------------------------------------------------------
            # Determine available API
            # -------------------------------------------------------------

            if hasattr(agent, "predict"):

                result = agent.predict(input_data)

            elif hasattr(agent, "run"):

                result = agent.run(input_data)

            elif hasattr(agent, "analyze"):

                result = agent.analyze(input_data)

            else:

                raise AttributeError(
                    f"{agent_name} has no predict(), run() or analyze() method."
                )

            latency = (
                datetime.now() - start
            ).total_seconds() * 1000.0

            # -------------------------------------------------------------
            # Normalize result
            # -------------------------------------------------------------

            if isinstance(result, AgentResult):

                output = result.to_dict()

            elif isinstance(result, dict):

                output = dict(result)

            else:

                output = {
                    "prediction": result
                }

            # -------------------------------------------------------------
            # Required metadata
            # -------------------------------------------------------------

            output.setdefault("agent_id", agent_name)
            output.setdefault("task_type", agent_name)
            output.setdefault("status", "success")

            output.setdefault("prediction", None)

            output.setdefault(
                "probability",
                output.get("confidence", 0.0)
            )

            output.setdefault(
                "confidence",
                output.get("probability", 0.0)
            )

            output.setdefault(
                "uncertainty",
                1.0 - float(output.get("confidence", 0.0))
            )

            output.setdefault("quality", 1.0)
            output.setdefault("missing_data_ratio", 0.0)

            output["latency_ms"] = float(
                output.get("latency_ms", latency)
            )

            # Preserve modality information for fusion
            output.setdefault("modality", agent_name)

            return output

        except Exception as e:

            latency = (
                datetime.now() - start
            ).total_seconds() * 1000.0

            return {
                "agent_id": agent_name,
                "task_type": agent_name,
                "status": "error",
                "prediction": None,
                "probability": 0.0,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "missing_data_ratio": 1.0,
                "latency_ms": latency,
                "modality": agent_name,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    # =========================================================================
    # TRUST
    # =========================================================================

    def _compute_trust(self, result: Dict[str, Any]) -> float:

        if result.get("status") != "success":
            return 0.0

        confidence = float(
            result.get("confidence", 0.0)
        )

        uncertainty = float(
            result.get("uncertainty", 1.0)
        )

        quality = float(
            result.get("quality", 1.0)
        )

        missing_ratio = float(
            result.get("missing_data_ratio", 0.0)
        )

        if self.trust_manager is None:
            return max(
                0.0,
                min(
                    1.0,
                    confidence * quality * (1.0 - uncertainty)
                )
            )

        try:

            trust = self.trust_manager.compute_trust(
                agent_id=result.get("agent_id", "unknown"),
                confidence=confidence,
                quality=quality,
                uncertainty=uncertainty,
                missing_data_ratio=missing_ratio,
                agreement=0.5,
                stability=0.5,
                utility=0.5,
                modality_available=True,
            )

            return float(trust)

        except Exception:

            return max(
                0.0,
                min(
                    1.0,
                    confidence * quality * (1.0 - uncertainty)
                )
            )

    # =========================================================================
    # CONVERSION TO AgentResult
    # =========================================================================

    def _make_agent_result(
        self,
        raw_result: Dict[str, Any],
    ) -> AgentResult:

        """
        Convert normalized dictionaries to AgentResult.

        The AgentResult object is used by:
            - Trust / conflict components

        The original dictionary is kept separately for:
            - adaptive fusion
            - metadata preservation
        """

        allowed = {
            "agent_id",
            "task_type",
            "prediction",
            "probability",
            "confidence",
            "uncertainty",
            "quality",
            "latency_ms",
            "missing_data_ratio",
            "trust",
            "status",
            "details",
            "explanation",
            "error",
        }

        data = {
            key: value
            for key, value in raw_result.items()
            if key in allowed
        }

        return AgentResult.from_dict(data)

    # =========================================================================
    # CLINICAL REASONING
    # =========================================================================

    def _run_clinical_reasoning(
        self,
        clinical_input: Any,
    ) -> Dict[str, Any]:

        if self.clinical_agent is None:

            return {
                "agent_id": "clinical_reasoning",
                "task_type": "clinical_reasoning",
                "status": "not_available",
                "prediction": None,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "missing_data_ratio": 1.0,
            }

        if not self._has_value(clinical_input):

            return {
                "agent_id": "clinical_reasoning",
                "task_type": "clinical_reasoning",
                "status": "not_run",
                "prediction": None,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "missing_data_ratio": 1.0,
                "error": "Clinical input not provided.",
            }

        return self._execute_agent(
            "clinical_reasoning",
            self.clinical_agent,
            clinical_input,
        )

    # =========================================================================
    # MAIN RUN METHOD
    # =========================================================================

    def run(
        self,
        patient_id: str,
        patient_data: Optional[Dict[str, Any]] = None,
        clinical_data: Optional[Dict[str, Any]] = None,
        fibrosis_input: Optional[Dict[str, Any]] = None,
        cirrhosis_input: Optional[Dict[str, Any]] = None,
        image: Any = None,
        volume: Any = None,
        clinical_reasoning_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        """
        Run the complete LiverAI multi-agent pipeline.

        Parameters
        ----------
        patient_id:
            Unique patient identifier.

        patient_data:
            Unified nested input dictionary.

        clinical_data:
            Legacy input for clinical reasoning.

        fibrosis_input:
            Legacy fibrosis input.

        cirrhosis_input:
            Legacy cirrhosis input.

        image:
            2D tumor image.

        volume:
            3D liver volume.

        clinical_reasoning_input:
            Explicit clinical reasoning input.

        Returns
        -------
        dict
            Unified LiverAI report.
        """

        timestamp = datetime.now().isoformat()

        # =====================================================================
        # Normalize top-level input
        # =====================================================================

        if patient_data is None:
            patient_data = {}

        if not isinstance(patient_data, dict):
            raise TypeError(
                "patient_data must be a dictionary."
            )

        # =====================================================================
        # Extract each modality
        # =====================================================================

        fatty_input = self._extract_modality(
            patient_data,
            "fatty_liver",
        )

        fibrosis_input_final = self._extract_modality(
            patient_data,
            "fibrosis",
            fibrosis_input,
        )

        cirrhosis_input_final = self._extract_modality(
            patient_data,
            "cirrhosis",
            cirrhosis_input,
        )

        tumor_input = self._extract_modality(
            patient_data,
            "tumor",
            image,
        )

        segmentation_input = self._extract_modality(
            patient_data,
            "segmentation",
            volume,
        )

        clinical_input = self._extract_modality(
            patient_data,
            "clinical_reasoning",
            clinical_reasoning_input,
        )

        if clinical_input is None:
            clinical_input = clinical_data

        # =====================================================================
        # Execute specialized agents
        # =====================================================================

        raw_results = {}

        raw_results["fatty_liver"] = self._execute_agent(
            "fatty_liver",
            self.fatty_liver_agent,
            fatty_input,
        )

        raw_results["fibrosis"] = self._execute_agent(
            "fibrosis",
            self.fibrosis_agent,
            fibrosis_input_final,
        )

        raw_results["cirrhosis"] = self._execute_agent(
            "cirrhosis",
            self.cirrhosis_agent,
            cirrhosis_input_final,
        )

        raw_results["tumor_classification"] = self._execute_agent(
            "tumor_classification",
            self.tumor_agent,
            tumor_input,
        )

        raw_results["liver_segmentation"] = self._execute_agent(
            "liver_segmentation",
            self.segmentation_agent,
            segmentation_input,
        )

        # =====================================================================
        # Clinical reasoning
        # =====================================================================

        clinical_result = self._run_clinical_reasoning(
            clinical_input
        )

        raw_results["clinical_reasoning"] = clinical_result

        # =====================================================================
        # Compute trust for every result
        # =====================================================================

        for name, result in raw_results.items():

            result["trust"] = self._compute_trust(result)

        # =====================================================================
        # Build AgentResult objects
        # =====================================================================

        agent_objects = []

        for result in raw_results.values():

            try:

                agent_objects.append(
                    self._make_agent_result(result)
                )

            except Exception as e:

                print(
                    f"Warning: could not create AgentResult for "
                    f"{result.get('agent_id')}: {e}"
                )

        # =====================================================================
        # Adaptive fusion
        # =====================================================================

        fusion_result = {}

        successful_specialized = []

        for name in [
            "fatty_liver",
            "fibrosis",
            "cirrhosis",
            "tumor_classification",
            "liver_segmentation",
        ]:

            result = raw_results[name]

            if result.get("status") == "success":
                successful_specialized.append(result)

        if self.adaptive_fusion is not None:

            try:

                fusion_result = self.adaptive_fusion.fuse(
                    successful_specialized
                )

            except Exception as e:

                fusion_result = {
                    "status": "error",
                    "error": str(e),
                }

        else:

            fusion_result = {
                "status": "not_available",
                "message": "AdaptiveFusion module is not available.",
            }

        # =====================================================================
        # Conflict detection
        # =====================================================================

        conflicts = []

        if self.conflict_detector is not None:

            try:

                conflicts = self.conflict_detector.detect(
                    agent_objects
                )

            except Exception as e:

                conflicts = [{
                    "status": "error",
                    "error": str(e),
                }]

        # =====================================================================
        # Final decision
        # =====================================================================

        decision = {}

        if self.decision_engine is not None:

            try:

                decision = self.decision_engine.decide(
                    list(raw_results.values()),
                    conflicts,
                    clinical_result,
                )

            except Exception as e:

                decision = {
                    "status": "error",
                    "error": str(e),
                }

        else:

            decision = {
                "status": "not_available",
                "message": "DecisionEngine module is not available.",
            }

        # =====================================================================
        # Execution statistics
        # =====================================================================

        completed_agents = [
            name
            for name, result in raw_results.items()
            if result.get("status") == "success"
        ]

        not_run_agents = [
            name
            for name, result in raw_results.items()
            if result.get("status") == "not_run"
        ]

        failed_agents = [
            name
            for name, result in raw_results.items()
            if result.get("status") in [
                "error",
                "not_available",
            ]
        ]

        # =====================================================================
        # Determine global status
        # =====================================================================

        if len(completed_agents) == 6:

            global_status = "completed"

        elif len(completed_agents) > 0:

            global_status = "partial"

        else:

            global_status = "failed"

        # =====================================================================
        # Input availability
        # =====================================================================

        input_availability = {

            "fatty_liver": self._has_value(fatty_input),

            "fibrosis": self._has_value(
                fibrosis_input_final
            ),

            "cirrhosis": self._has_value(
                cirrhosis_input_final
            ),

            "tumor": self._has_value(
                tumor_input
            ),

            "segmentation": self._has_value(
                segmentation_input
            ),

            "clinical_reasoning": self._has_value(
                clinical_input
            ),
        }

        # =====================================================================
        # FINAL UNIFIED REPORT
        # =====================================================================

        return {

            "system": self.name,

            "patient_id": patient_id,

            "timestamp": timestamp,

            "status": global_status,

            "total_specialized_agents": 5,

            "total_agents": 6,

            "agents_completed": len(
                completed_agents
            ),

            "completed_agent_names": completed_agents,

            "not_run_agents": not_run_agents,

            "failed_agents": failed_agents,

            "input_availability": input_availability,

            "agents": raw_results,

            "adaptive_fusion": fusion_result,

            "conflicts": conflicts,

            "clinical_reasoning": clinical_result,

            "decision": decision,

            "execution": {

                "successful_agents":
                    len(completed_agents),

                "not_run_agents":
                    len(not_run_agents),

                "failed_agents":
                    len(failed_agents),

                "coverage":
                    len(completed_agents) / 6.0,
            },

            "note": (
                "The decision and risk values are system-level "
                "AI coordination outputs and must not be interpreted "
                "as a standalone clinical diagnosis."
            ),
        }


# =============================================================================
# END OF FILE
# =============================================================================
