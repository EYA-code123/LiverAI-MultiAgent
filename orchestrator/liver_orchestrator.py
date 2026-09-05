# =============================================================================
# LiverAI-MultiAgent
# FILE: orchestrator/liver_orchestrator.py
# COMPLETE ADAPTIVE MULTI-AGENT ORCHESTRATOR
# =============================================================================

from typing import Dict, Any, Optional
from datetime import datetime
import time
import traceback
import joblib


# =============================================================================
# AGENT IMPORTS
# =============================================================================

try:
    from agents.fatty_liver_agent import FattyLiverAgent
except Exception as e:
    FattyLiverAgent = None
    print("WARNING: FattyLiverAgent import failed:", repr(e))


try:
    from agents.fibrosis_agent import FibrosisAgent
except Exception as e:
    FibrosisAgent = None
    print("WARNING: FibrosisAgent import failed:", repr(e))


try:
    from agents.cirrhosis_agent import CirrhosisAgent
except Exception as e:
    CirrhosisAgent = None
    print("WARNING: CirrhosisAgent import failed:", repr(e))


try:
    from agents.tumor_classification_agent import TumorClassificationAgent
except Exception as e:
    TumorClassificationAgent = None
    print("WARNING: TumorClassificationAgent import failed:", repr(e))


try:
    from agents.liver_segmentation_agent import LiverSegmentationAgent
except Exception as e:
    LiverSegmentationAgent = None
    print("WARNING: LiverSegmentationAgent import failed:", repr(e))


try:
    from agents.clinical_reasoning_agent import ClinicalReasoningAgent
except Exception as e:
    ClinicalReasoningAgent = None
    print("WARNING: ClinicalReasoningAgent import failed:", repr(e))


# =============================================================================
# COORDINATION IMPORTS
# =============================================================================

try:
    from orchestrator.schemas import AgentResult
except Exception as e:
    AgentResult = None
    print("WARNING: AgentResult import failed:", repr(e))


try:
    from coordinator.trust import TrustManager
except Exception as e:
    TrustManager = None
    print("WARNING: TrustManager import failed:", repr(e))


try:
    from coordinator.adaptive_fusion import AdaptiveFusion
except Exception as e:
    AdaptiveFusion = None
    print("WARNING: AdaptiveFusion import failed:", repr(e))


try:
    from coordinator.conflict import ConflictDetector
except Exception as e:
    ConflictDetector = None
    print("WARNING: ConflictDetector import failed:", repr(e))


try:
    from coordinator.decision import DecisionEngine
except Exception as e:
    DecisionEngine = None
    print("WARNING: DecisionEngine import failed:", repr(e))


# =============================================================================
# MODEL PATHS
# =============================================================================

FATTY_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "Fatty_Liver_Dataset/models/"
    "FattyLiver_LightGBM.pkl"
)

FIBROSIS_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "Fibrosis Agent/XGBoost_model/"
    "xgboost_nafld.pkl"
)

CIRRHOSIS_MODEL_PATH = (
    "/content/drive/MyDrive/"
    ".Cirrhosis Agent/XGBoost_model/"
    "XGBoost_Cirrhosis_fixed.joblib"
)

TUMOR_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "models/tumor/"
    "efficientnet_b0_best.pth"
)

SEGMENTATION_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "Liver Segmentation Agent/models/"
    "SegResNet3D_Liver_best.pth"
)

CLINICAL_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "Clinical Reasoning Agent/"
    "tabtransformer_bupa"
)


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class LiverAIOrchestrator:
    """
    Central coordination layer for the six LiverAI agents.

    Agents
    ------
    1. Fatty Liver
    2. Fibrosis
    3. Cirrhosis
    4. Tumor Classification
    5. Liver Segmentation
    6. Clinical Reasoning

    Pipeline
    --------
    Patient Data
        ↓
    Specialized Agents
        ↓
    Standardization
        ↓
    Trust Evaluation
        ↓
    Adaptive Evidence Fusion
        ↓
    Conflict Detection
        ↓
    Clinical Reasoning
        ↓
    Decision Intelligence
        ↓
    Final Coordinated Assessment
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        fatty_agent=None,
        fibrosis_agent=None,
        cirrhosis_agent=None,
        tumor_agent=None,
        segmentation_agent=None,
        clinical_reasoning_agent=None,

        # Compatibility aliases
        fatty_liver_agent=None,
        tumor_classification_agent=None,
        liver_segmentation_agent=None,
        clinical_agent=None,

        auto_initialize=True,
        device=None,
    ):

        self.name = "LiverAI Adaptive Multi-Agent Orchestrator"

        # ---------------------------------------------------------------------
        # DEVICE
        # ---------------------------------------------------------------------

        if device is None:
            try:
                import torch

                self.device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                )

            except Exception:
                self.device = "cpu"

        else:
            self.device = device

        # ---------------------------------------------------------------------
        # INITIALIZATION ERRORS
        # ---------------------------------------------------------------------

        self.initialization_errors = {}

        # ---------------------------------------------------------------------
        # AGENTS
        # ---------------------------------------------------------------------

        self.fatty_agent = (
            fatty_agent
            if fatty_agent is not None
            else fatty_liver_agent
        )

        self.fibrosis_agent = fibrosis_agent

        self.cirrhosis_agent = cirrhosis_agent

        self.tumor_agent = (
            tumor_agent
            if tumor_agent is not None
            else tumor_classification_agent
        )

        self.segmentation_agent = (
            segmentation_agent
            if segmentation_agent is not None
            else liver_segmentation_agent
        )

        self.clinical_reasoning_agent = (
            clinical_reasoning_agent
            if clinical_reasoning_agent is not None
            else clinical_agent
        )

        # ---------------------------------------------------------------------
        # COORDINATION MODULES
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
        # AUTOMATIC INITIALIZATION
        # ---------------------------------------------------------------------

        if auto_initialize:
            self._initialize_agents()

        # ---------------------------------------------------------------------
        # REGISTRY
        # ---------------------------------------------------------------------

        self.agent_registry = {
            "fatty_liver": self.fatty_agent,
            "fibrosis": self.fibrosis_agent,
            "cirrhosis": self.cirrhosis_agent,
            "tumor_classification": self.tumor_agent,
            "liver_segmentation": self.segmentation_agent,
            "clinical_reasoning": self.clinical_reasoning_agent,
        }

        # ---------------------------------------------------------------------
        # STATE
        # ---------------------------------------------------------------------

        self.last_results = {}
        self.last_assessment = None
        self.last_final_decision = None
        self.execution_log = []

    # =========================================================================
    # INITIALIZE AGENTS
    # =========================================================================

    def _initialize_agents(self):

        print("=" * 70)
        print("LIVER AI AGENTS INITIALIZATION")
        print("=" * 70)

        # ---------------------------------------------------------------------
        # 1. FATTY LIVER
        # ---------------------------------------------------------------------

        if self.fatty_agent is None:

            try:

                if FattyLiverAgent is None:
                    raise ImportError(
                        "FattyLiverAgent could not be imported."
                    )

                fatty_model_package = joblib.load(
                    FATTY_MODEL_PATH
                )

                self.fatty_agent = FattyLiverAgent(
                    fatty_model_package
                )

                print("✓ Fatty Liver Agent initialized")

            except Exception as e:

                self.initialization_errors[
                    "fatty_liver"
                ] = repr(e)

                print(
                    "❌ Fatty Liver Agent:",
                    repr(e)
                )

        else:
            print("✓ Fatty Liver Agent supplied externally")

        # ---------------------------------------------------------------------
        # 2. FIBROSIS
        # ---------------------------------------------------------------------

        if self.fibrosis_agent is None:

            try:

                if FibrosisAgent is None:
                    raise ImportError(
                        "FibrosisAgent could not be imported."
                    )

                fibrosis_model = joblib.load(
                    FIBROSIS_MODEL_PATH
                )

                self.fibrosis_agent = FibrosisAgent(
                    fibrosis_model
                )

                print("✓ Fibrosis Agent initialized")

            except Exception as e:

                self.initialization_errors[
                    "fibrosis"
                ] = repr(e)

                print(
                    "❌ Fibrosis Agent:",
                    repr(e)
                )

        else:
            print("✓ Fibrosis Agent supplied externally")

        # ---------------------------------------------------------------------
        # 3. CIRRHOSIS
        # ---------------------------------------------------------------------

        if self.cirrhosis_agent is None:

            try:

                if CirrhosisAgent is None:
                    raise ImportError(
                        "CirrhosisAgent could not be imported."
                    )

                self.cirrhosis_agent = CirrhosisAgent(
                    CIRRHOSIS_MODEL_PATH
                )

                print("✓ Cirrhosis Agent initialized")

            except Exception as e:

                self.initialization_errors[
                    "cirrhosis"
                ] = repr(e)

                print(
                    "❌ Cirrhosis Agent:",
                    repr(e)
                )

        else:
            print("✓ Cirrhosis Agent supplied externally")

        # ---------------------------------------------------------------------
        # 4. TUMOR CLASSIFICATION
        # ---------------------------------------------------------------------

        if self.tumor_agent is None:

            try:

                if TumorClassificationAgent is None:
                    raise ImportError(
                        "TumorClassificationAgent "
                        "could not be imported."
                    )

                self.tumor_agent = TumorClassificationAgent(
                    TUMOR_MODEL_PATH
                )

                print(
                    "✓ Tumor Classification Agent initialized"
                )

            except Exception as e:

                self.initialization_errors[
                    "tumor_classification"
                ] = repr(e)

                print(
                    "❌ Tumor Classification Agent:",
                    repr(e)
                )

        else:
            print(
                "✓ Tumor Classification Agent supplied externally"
            )

        # ---------------------------------------------------------------------
        # 5. LIVER SEGMENTATION
        # ---------------------------------------------------------------------

        if self.segmentation_agent is None:

            try:

                if LiverSegmentationAgent is None:
                    raise ImportError(
                        "LiverSegmentationAgent "
                        "could not be imported."
                    )

                self.segmentation_agent = (
                    LiverSegmentationAgent(
                        model_path=SEGMENTATION_MODEL_PATH,
                        device=self.device,
                        target_size=(128, 128, 64),
                        threshold=0.5,
                    )
                )

                print(
                    "✓ Liver Segmentation Agent initialized"
                )

            except Exception as e:

                self.initialization_errors[
                    "liver_segmentation"
                ] = repr(e)

                print(
                    "❌ Liver Segmentation Agent:",
                    repr(e)
                )

        else:
            print(
                "✓ Liver Segmentation Agent supplied externally"
            )

        # ---------------------------------------------------------------------
        # 6. CLINICAL REASONING
        # ---------------------------------------------------------------------

        if self.clinical_reasoning_agent is None:

            try:

                if ClinicalReasoningAgent is None:
                    raise ImportError(
                        "ClinicalReasoningAgent "
                        "could not be imported."
                    )

                self.clinical_reasoning_agent = (
                    ClinicalReasoningAgent(
                        CLINICAL_MODEL_PATH
                    )
                )

                print(
                    "✓ Clinical Reasoning Agent initialized"
                )

            except Exception as e:

                self.initialization_errors[
                    "clinical_reasoning"
                ] = repr(e)

                print(
                    "❌ Clinical Reasoning Agent:",
                    repr(e)
                )

        else:
            print(
                "✓ Clinical Reasoning Agent supplied externally"
            )

        # ---------------------------------------------------------------------
        # UPDATE REGISTRY
        # ---------------------------------------------------------------------

        self.agent_registry = {
            "fatty_liver": self.fatty_agent,
            "fibrosis": self.fibrosis_agent,
            "cirrhosis": self.cirrhosis_agent,
            "tumor_classification": self.tumor_agent,
            "liver_segmentation": self.segmentation_agent,
            "clinical_reasoning": self.clinical_reasoning_agent,
        }

        # ---------------------------------------------------------------------
        # SUMMARY
        # ---------------------------------------------------------------------

        initialized = sum(
            agent is not None
            for agent in [
                self.fatty_agent,
                self.fibrosis_agent,
                self.cirrhosis_agent,
                self.tumor_agent,
                self.segmentation_agent,
                self.clinical_reasoning_agent,
            ]
        )

        print("=" * 70)
        print(
            f"INITIALIZED AGENTS: {initialized}/6"
        )

        if initialized == 6:
            print(
                "✓ ALL SIX AGENTS INITIALIZED SUCCESSFULLY"
            )

        else:
            print(
                "⚠ SOME AGENTS FAILED TO INITIALIZE"
            )

            for name, error in (
                self.initialization_errors.items()
            ):
                print(
                    f"  - {name}: {error}"
                )

        print("=" * 70)

    # =========================================================================
    # MAIN RUN
    # =========================================================================

    def run(
        self,
        patient_id: str = "UNKNOWN",
        patient_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        start_time = time.perf_counter()

        if patient_data is None:
            patient_data = {}

        # ---------------------------------------------------------------------
        # STEP 1 - SPECIALIZED AGENTS
        # ---------------------------------------------------------------------

        specialized_results = (
            self.run_specialized_agents(
                patient_data
            )
        )

        # ---------------------------------------------------------------------
        # STEP 2 - CLINICAL REASONING
        # ---------------------------------------------------------------------

        clinical_reasoning = (
            self.run_clinical_reasoning(
                patient_data,
                specialized_results,
            )
        )

        # ---------------------------------------------------------------------
        # STEP 3 - ALL RESULTS
        # ---------------------------------------------------------------------

        all_results = dict(
            specialized_results
        )

        all_results[
            "clinical_reasoning"
        ] = clinical_reasoning

        # ---------------------------------------------------------------------
        # STEP 4 - FUSION
        # ---------------------------------------------------------------------

        fusion = self._run_adaptive_fusion(
            all_results
        )

        # ---------------------------------------------------------------------
        # STEP 5 - CONFLICT DETECTION
        # ---------------------------------------------------------------------

        conflicts = self._run_conflict_detection(
            all_results
        )

        # ---------------------------------------------------------------------
        # STEP 6 - DECISION ENGINE
        # ---------------------------------------------------------------------

        decision = self._run_decision_engine(
            results=all_results,
            conflicts=conflicts,
            fusion=fusion,
            clinical_result=clinical_reasoning,
        )

        # ---------------------------------------------------------------------
        # STEP 7 - FINAL DECISION
        # ---------------------------------------------------------------------

        final_decision = self._build_final_decision(
            decision=decision,
            fusion=fusion,
            conflicts=conflicts,
            clinical_reasoning=clinical_reasoning,
            all_results=all_results,
        )

        # ---------------------------------------------------------------------
        # STATUS
        # ---------------------------------------------------------------------

        successful_agents = sum(
            1
            for result in all_results.values()
            if result.get("status") == "success"
        )

        total_agents = 6

        if successful_agents == 0:
            overall_status = "failed"

        elif successful_agents == total_agents:
            overall_status = "success"

        else:
            overall_status = "partial"

        # ---------------------------------------------------------------------
        # COORDINATION
        # ---------------------------------------------------------------------

        failed_agents = [
            name
            for name, result in all_results.items()
            if result.get("status") == "error"
        ]

        not_run_agents = [
            name
            for name, result in all_results.items()
            if result.get("status") == "not_run"
        ]

        coverage = (
            successful_agents / total_agents
            if total_agents > 0
            else 0.0
        )

        # ---------------------------------------------------------------------
        # LATENCY
        # ---------------------------------------------------------------------

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        # ---------------------------------------------------------------------
        # CLINICAL CONTEXT
        # ---------------------------------------------------------------------

        clinical_context = (
            self._build_clinical_context(
                all_results
            )
        )

        # ---------------------------------------------------------------------
        # SAVE STATE
        # ---------------------------------------------------------------------

        self.last_results = all_results
        self.last_final_decision = final_decision

        # ---------------------------------------------------------------------
        # FINAL OUTPUT
        # ---------------------------------------------------------------------

        return {
            "status": overall_status,

            "patient_id": patient_id,

            "timestamp": datetime.now().isoformat(),

            # -------------------------------------------------------------
            # AGENT RESULTS
            # -------------------------------------------------------------

            "agent_results": all_results,

            "specialized_results": specialized_results,

            "clinical_reasoning": clinical_reasoning,

            # -------------------------------------------------------------
            # COORDINATION
            # -------------------------------------------------------------

            "coordination": {
                "total_agents": total_agents,
                "successful_agents": successful_agents,
                "failed_agents": failed_agents,
                "not_run_agents": not_run_agents,
                "coverage": round(
                    coverage,
                    4
                ),
                "latency_ms": round(
                    latency_ms,
                    3
                ),
            },

            # -------------------------------------------------------------
            # FUSION
            # -------------------------------------------------------------

            "fusion": fusion,

            # -------------------------------------------------------------
            # CONFLICTS
            # -------------------------------------------------------------

            "conflicts": conflicts,

            # -------------------------------------------------------------
            # DECISION ENGINE
            # -------------------------------------------------------------

            "decision": decision,

            # -------------------------------------------------------------
            # FINAL DECISION
            # -------------------------------------------------------------

            "final_decision": final_decision,

            # -------------------------------------------------------------
            # CONTEXT
            # -------------------------------------------------------------

            "clinical_context": clinical_context,
        }

    # =========================================================================
    # ALIASES
    # =========================================================================

    def analyze(
        self,
        patient_id="UNKNOWN",
        patient_data=None,
    ):
        return self.run(
            patient_id=patient_id,
            patient_data=patient_data,
        )

    def predict(
        self,
        patient_id="UNKNOWN",
        patient_data=None,
    ):
        return self.run(
            patient_id=patient_id,
            patient_data=patient_data,
        )

    # =========================================================================
    # SPECIALIZED AGENTS
    # =========================================================================

    def run_specialized_agents(
        self,
        patient_data: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:

        results = {}

        # ---------------------------------------------------------------------
        # FATTY LIVER
        # ---------------------------------------------------------------------

        fatty_input = patient_data.get(
            "fatty_liver"
        )

        if fatty_input is not None:

            results["fatty_liver"] = (
                self._execute_agent(
                    agent_name="fatty_liver",
                    agent=self.fatty_agent,
                    input_data=fatty_input,
                )
            )

        else:

            results["fatty_liver"] = (
                self._not_run_result(
                    "fatty_liver",
                    "No fatty liver input provided.",
                )
            )

        # ---------------------------------------------------------------------
        # FIBROSIS
        # ---------------------------------------------------------------------

        fibrosis_input = patient_data.get(
            "fibrosis"
        )

        if fibrosis_input is not None:

            results["fibrosis"] = (
                self._execute_agent(
                    agent_name="fibrosis",
                    agent=self.fibrosis_agent,
                    input_data=fibrosis_input,
                )
            )

        else:

            results["fibrosis"] = (
                self._not_run_result(
                    "fibrosis",
                    "No fibrosis input provided.",
                )
            )

        # ---------------------------------------------------------------------
        # CIRRHOSIS
        # ---------------------------------------------------------------------

        cirrhosis_input = patient_data.get(
            "cirrhosis"
        )

        if cirrhosis_input is not None:

            results["cirrhosis"] = (
                self._execute_agent(
                    agent_name="cirrhosis",
                    agent=self.cirrhosis_agent,
                    input_data=cirrhosis_input,
                )
            )

        else:

            results["cirrhosis"] = (
                self._not_run_result(
                    "cirrhosis",
                    "No cirrhosis input provided.",
                )
            )

        # ---------------------------------------------------------------------
        # TUMOR
        # ---------------------------------------------------------------------

        tumor_input = patient_data.get(
            "tumor"
        )

        if tumor_input is None:
            tumor_input = patient_data.get(
                "tumor_classification"
            )

        if tumor_input is not None:

            results["tumor_classification"] = (
                self._execute_agent(
                    agent_name="tumor_classification",
                    agent=self.tumor_agent,
                    input_data=tumor_input,
                )
            )

        else:

            results["tumor_classification"] = (
                self._not_run_result(
                    "tumor_classification",
                    "No tumor image provided.",
                )
            )

        # ---------------------------------------------------------------------
        # LIVER SEGMENTATION
        # ---------------------------------------------------------------------

        segmentation_input = patient_data.get(
            "segmentation"
        )

        if segmentation_input is None:
            segmentation_input = patient_data.get(
                "liver_segmentation"
            )

        if segmentation_input is not None:

            results["liver_segmentation"] = (
                self._execute_agent(
                    agent_name="liver_segmentation",
                    agent=self.segmentation_agent,
                    input_data=segmentation_input,
                )
            )

        else:

            results["liver_segmentation"] = (
                self._not_run_result(
                    "liver_segmentation",
                    "No liver volume provided.",
                )
            )

        return results

    # =========================================================================
    # CLINICAL REASONING
    # =========================================================================

    def run_clinical_reasoning(
        self,
        patient_data: Dict[str, Any],
        specialized_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        if self.clinical_reasoning_agent is None:

            return self._not_run_result(
                "clinical_reasoning",
                "Clinical reasoning agent not initialized.",
            )

        # ---------------------------------------------------------------------
        # CLINICAL MODEL INPUT
        # ---------------------------------------------------------------------

        clinical_input = patient_data.get(
            "clinical_reasoning"
        )

        if clinical_input is None:
            clinical_input = patient_data.get(
                "fatty_liver"
            )

        if clinical_input is None:

            return self._not_run_result(
                "clinical_reasoning",
                "No clinical reasoning input provided.",
            )

        # ---------------------------------------------------------------------
        # EXECUTION
        # ---------------------------------------------------------------------

        result = self._execute_agent(
            agent_name="clinical_reasoning",
            agent=self.clinical_reasoning_agent,
            input_data=clinical_input,
        )

        # ---------------------------------------------------------------------
        # SPECIALIZED EVIDENCE
        # ---------------------------------------------------------------------

        result["specialized_evidence"] = {}

        for name, agent_result in (
            specialized_results.items()
        ):

            if agent_result.get("status") == "success":

                result[
                    "specialized_evidence"
                ][name] = {
                    "prediction": agent_result.get(
                        "prediction"
                    ),
                    "confidence": agent_result.get(
                        "confidence"
                    ),
                    "trust": agent_result.get(
                        "trust"
                    ),
                    "probability": agent_result.get(
                        "probability"
                    ),
                }

        return result

    # =========================================================================
    # EXECUTE AGENT
    # =========================================================================

    def _execute_agent(
        self,
        agent_name: str,
        agent,
        input_data: Any,
    ) -> Dict[str, Any]:

        if agent is None:

            return self._not_run_result(
                agent_name,
                "Agent not initialized.",
            )

        start = time.perf_counter()

        try:

            # -----------------------------------------------------------------
            # FIND EXECUTION METHOD
            # -----------------------------------------------------------------

            if hasattr(agent, "predict"):

                raw_result = agent.predict(
                    input_data
                )

            elif hasattr(agent, "run"):

                raw_result = agent.run(
                    input_data
                )

            elif hasattr(agent, "analyze"):

                raw_result = agent.analyze(
                    input_data
                )

            else:

                raise AttributeError(
                    f"{type(agent).__name__} "
                    "has no predict(), run(), "
                    "or analyze() method."
                )

            latency_ms = (
                time.perf_counter() - start
            ) * 1000.0

            return self._normalize_result(
                agent_name=agent_name,
                result=raw_result,
                latency_ms=latency_ms,
            )

        except Exception as e:

            latency_ms = (
                time.perf_counter() - start
            ) * 1000.0

            return {
                "agent_id": agent_name,
                "agent": agent_name,
                "task_type": agent_name,
                "status": "error",
                "prediction": None,
                "probability": None,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "trust": 0.0,
                "latency_ms": round(
                    latency_ms,
                    3,
                ),
                "missing_data_ratio": 0.0,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }

    # =========================================================================
    # NORMALIZE RESULT
    # =========================================================================

    def _normalize_result(
        self,
        agent_name: str,
        result: Any,
        latency_ms: float = 0.0,
    ) -> Dict[str, Any]:

        if result is None:

            return {
                "agent_id": agent_name,
                "agent": agent_name,
                "task_type": agent_name,
                "status": "error",
                "prediction": None,
                "probability": None,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "trust": 0.0,
                "latency_ms": round(
                    latency_ms,
                    3,
                ),
                "missing_data_ratio": 1.0,
                "error": "Agent returned None.",
            }

        # ---------------------------------------------------------------------
        # DICTIONARY RESULT
        # ---------------------------------------------------------------------

        if isinstance(result, dict):

            normalized = dict(result)

            normalized.setdefault(
                "agent_id",
                agent_name,
            )

            normalized.setdefault(
                "agent",
                agent_name,
            )

            normalized.setdefault(
                "task_type",
                agent_name,
            )

            normalized.setdefault(
                "status",
                "success",
            )

            normalized.setdefault(
                "latency_ms",
                round(
                    latency_ms,
                    3,
                ),
            )

            normalized.setdefault(
                "missing_data_ratio",
                0.0,
            )

            normalized.setdefault(
                "prediction",
                None,
            )

            # -----------------------------------------------------------------
            # CONFIDENCE
            # -----------------------------------------------------------------

            confidence = self._extract_confidence(
                normalized
            )

            normalized[
                "confidence"
            ] = confidence

            # -----------------------------------------------------------------
            # UNCERTAINTY
            # -----------------------------------------------------------------

            uncertainty = normalized.get(
                "uncertainty"
            )

            if uncertainty is None:

                uncertainty = (
                    1.0 - confidence
                )

            normalized[
                "uncertainty"
            ] = self._clip(
                uncertainty
            )

            # -----------------------------------------------------------------
            # QUALITY
            # -----------------------------------------------------------------

            quality = normalized.get(
                "quality"
            )

            if quality is None:
                quality = 1.0

            normalized[
                "quality"
            ] = self._clip(
                quality
            )

            # -----------------------------------------------------------------
            # TRUST
            # -----------------------------------------------------------------

            if normalized.get("status") == "success":

                normalized[
                    "trust"
                ] = self._compute_trust(
                    agent_name,
                    normalized,
                )

            else:

                normalized[
                    "trust"
                ] = 0.0

            # -----------------------------------------------------------------
            # MODALITY
            # -----------------------------------------------------------------

            if "modality" not in normalized:

                if agent_name in [
                    "fatty_liver",
                    "fibrosis",
                    "cirrhosis",
                    "clinical_reasoning",
                ]:

                    normalized[
                        "modality"
                    ] = "tabular"

                elif agent_name == (
                    "tumor_classification"
                ):

                    normalized[
                        "modality"
                    ] = "2D_image"

                elif agent_name == (
                    "liver_segmentation"
                ):

                    normalized[
                        "modality"
                    ] = "3D_volume"

            return normalized

        # ---------------------------------------------------------------------
        # NON-DICTIONARY RESULT
        # ---------------------------------------------------------------------

        confidence = 0.5

        normalized = {
            "agent_id": agent_name,
            "agent": agent_name,
            "task_type": agent_name,
            "status": "success",
            "prediction": result,
            "probability": None,
            "confidence": confidence,
            "uncertainty": 1.0 - confidence,
            "quality": 1.0,
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "missing_data_ratio": 0.0,
            "modality": "unknown",
        }

        normalized[
            "trust"
        ] = self._compute_trust(
            agent_name,
            normalized,
        )

        return normalized

    # =========================================================================
    # CONFIDENCE EXTRACTION
    # =========================================================================

    def _extract_confidence(
        self,
        result: Dict[str, Any],
    ) -> float:

        confidence = result.get(
            "confidence"
        )

        if confidence is not None:

            value = self._safe_float(
                confidence,
                default=None,
            )

            if value is not None:
                return self._clip(value)

        probability = result.get(
            "probability"
        )

        if probability is not None:

            value = self._probability_confidence(
                probability
            )

            if value is not None:
                return self._clip(value)

        class_probabilities = result.get(
            "class_probabilities"
        )

        if class_probabilities is not None:

            value = self._probability_confidence(
                class_probabilities
            )

            if value is not None:
                return self._clip(value)

        probabilities = result.get(
            "probabilities"
        )

        if probabilities is not None:

            value = self._probability_confidence(
                probabilities
            )

            if value is not None:
                return self._clip(value)

        return 0.5

    # =========================================================================
    # PROBABILITY CONFIDENCE
    # =========================================================================

    def _probability_confidence(
        self,
        probabilities,
    ):

        try:

            if isinstance(
                probabilities,
                dict,
            ):

                values = []

                for value in probabilities.values():

                    try:
                        values.append(
                            float(value)
                        )

                    except (
                        ValueError,
                        TypeError,
                    ):
                        continue

                if values:
                    return max(values)

                return None

            if isinstance(
                probabilities,
                (list, tuple)
            ):

                if len(probabilities) == 0:
                    return None

                values = [
                    float(x)
                    for x in probabilities
                ]

                return max(values)

            return float(
                probabilities
            )

        except (
            ValueError,
            TypeError,
        ):

            return None

    # =========================================================================
    # TRUST
    # =========================================================================

    def _compute_trust(
        self,
        agent_name: str,
        result: Dict[str, Any],
    ) -> float:

        if self.trust_manager is None:
            return 0.0

        confidence = self._clip(
            result.get(
                "confidence",
                0.0,
            )
        )

        quality = self._clip(
            result.get(
                "quality",
                0.0,
            )
        )

        uncertainty = self._clip(
            result.get(
                "uncertainty",
                1.0,
            )
        )

        missing_ratio = self._clip(
            result.get(
                "missing_data_ratio",
                0.0,
            )
        )

        try:

            trust = self.trust_manager.compute_trust(
                agent_id=agent_name,
                confidence=confidence,
                quality=quality,
                uncertainty=uncertainty,
                missing_data_ratio=missing_ratio,
            )

            return self._clip(trust)

        except TypeError:

            try:

                trust = self.trust_manager.compute_trust(
                    agent_name,
                    confidence,
                    quality,
                    uncertainty,
                    missing_ratio,
                )

                return self._clip(trust)

            except Exception:
                return 0.0

        except Exception:
            return 0.0

    # =========================================================================
    # ADAPTIVE FUSION
    # =========================================================================

    def _run_adaptive_fusion(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        if self.adaptive_fusion is None:

            return {
                "status": "unavailable",
                "evidence": [],
                "weights": {},
                "task_groups": {},
            }

        valid_results = [
            result
            for result in results.values()
            if result.get("status") == "success"
        ]

        if not valid_results:

            return {
                "status": "no_valid_results",
                "evidence": [],
                "weights": {},
                "task_groups": {},
                "same_task_fusion": {},
            }

        try:

            fusion = self.adaptive_fusion.fuse(
                valid_results
            )

            if isinstance(
                fusion,
                dict,
            ):
                return fusion

            return {
                "status": "success",
                "result": fusion,
            }

        except Exception as e:

            return {
                "status": "error",
                "error": repr(e),
                "evidence": valid_results,
            }

    # =========================================================================
    # CONFLICT DETECTION
    # =========================================================================

    def _run_conflict_detection(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> list:

        if self.conflict_detector is None:
            return []

        valid_results = [
            result
            for result in results.values()
            if result.get("status") == "success"
            and result.get("prediction") is not None
        ]

        if len(valid_results) < 2:
            return []

        if AgentResult is None:
            return []

        agent_objects = []

        for result in valid_results:

            try:

                agent_objects.append(
                    AgentResult.from_dict(
                        result
                    )
                )

            except Exception:

                try:

                    agent_objects.append(
                        AgentResult(
                            agent_id=result.get(
                                "agent_id",
                                "unknown",
                            ),
                            task_type=result.get(
                                "task_type",
                                "unknown",
                            ),
                            prediction=result.get(
                                "prediction"
                            ),
                            probability=result.get(
                                "probability"
                            ),
                            confidence=result.get(
                                "confidence",
                                0.0,
                            ),
                            uncertainty=result.get(
                                "uncertainty",
                                1.0,
                            ),
                            quality=result.get(
                                "quality",
                                0.0,
                            ),
                            latency_ms=result.get(
                                "latency_ms",
                                0.0,
                            ),
                            missing_data_ratio=result.get(
                                "missing_data_ratio",
                                0.0,
                            ),
                            trust=result.get(
                                "trust",
                                0.0,
                            ),
                            status=result.get(
                                "status",
                                "success",
                            ),
                            details=result.get(
                                "details",
                                {},
                            ),
                            explanation=result.get(
                                "explanation"
                            ),
                            error=result.get(
                                "error"
                            ),
                        )
                    )

                except Exception:
                    continue

        if len(agent_objects) < 2:
            return []

        try:

            conflicts = (
                self.conflict_detector.detect(
                    agent_objects
                )
            )

            if conflicts is None:
                return []

            return conflicts

        except Exception as e:

            return [
                {
                    "type": "conflict_detection_error",
                    "error": repr(e),
                }
            ]

    # =========================================================================
    # DECISION ENGINE
    # =========================================================================

    def _run_decision_engine(
        self,
        results: Dict[str, Dict[str, Any]],
        conflicts,
        fusion: Dict[str, Any],
        clinical_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        if self.decision_engine is None:

            return {
                "status": "unavailable",
                "decision": "insufficient_evidence",
                "confidence": 0.0,
                "risk_score": 1.0,
                "coverage": 0.0,
                "error": "DecisionEngine unavailable.",
            }

        all_results = list(
            results.values()
        )

        try:

            # =================================================================
            # IMPORTANT FIX
            #
            # Use KEYWORD arguments.
            #
            # Old incorrect version:
            #
            # decide(
            #     all_results,
            #     conflicts,
            #     clinical_result
            # )
            #
            # This sent clinical_result into fused_results.
            # =================================================================

            decision = self.decision_engine.decide(
                agent_results=all_results,
                conflicts=conflicts,
                fused_results=fusion,
                clinical_reasoning=clinical_result,
            )

            if isinstance(
                decision,
                dict,
            ):
                return decision

            return {
                "status": "completed",
                "decision": str(decision),
                "confidence": 0.0,
                "risk_score": None,
                "raw_result": decision,
            }

        except TypeError:

            # -----------------------------------------------------------------
            # Compatibility fallback for an older DecisionEngine
            # -----------------------------------------------------------------

            try:

                decision = self.decision_engine.decide(
                    all_results,
                    conflicts,
                    fusion,
                    clinical_result,
                )

                if isinstance(
                    decision,
                    dict,
                ):
                    return decision

                return {
                    "status": "completed",
                    "decision": str(decision),
                    "confidence": 0.0,
                    "risk_score": None,
                }

            except Exception as e:

                return {
                    "status": "error",
                    "decision": "insufficient_evidence",
                    "confidence": 0.0,
                    "risk_score": 1.0,
                    "coverage": 0.0,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }

        except Exception as e:

            return {
                "status": "error",
                "decision": "insufficient_evidence",
                "confidence": 0.0,
                "risk_score": 1.0,
                "coverage": 0.0,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }

    # =========================================================================
    # BUILD FINAL DECISION
    # =========================================================================

    def _build_final_decision(
        self,
        decision: Dict[str, Any],
        fusion: Dict[str, Any],
        conflicts,
        clinical_reasoning: Dict[str, Any],
        all_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not isinstance(
            decision,
            dict,
        ):
            decision = {}

        # ---------------------------------------------------------------------
        # DECISION LABEL
        # ---------------------------------------------------------------------

        decision_label = decision.get(
            "decision"
        )

        if decision_label is None:

            decision_label = decision.get(
                "status",
                "insufficient_evidence"
            )

        # ---------------------------------------------------------------------
        # CONFIDENCE
        # ---------------------------------------------------------------------

        confidence = decision.get(
            "confidence"
        )

        if confidence is None:

            confidence = decision.get(
                "decision_confidence",
                decision.get(
                    "clinical_confidence",
                    0.0,
                ),
            )

        confidence = self._clip(
            confidence
        )

        # ---------------------------------------------------------------------
        # RISK
        # ---------------------------------------------------------------------

        risk_score = decision.get(
            "risk_score"
        )

        if risk_score is None:

            risk_score = (
                1.0 - confidence
            )

        risk_score = self._clip(
            risk_score
        )

        # ---------------------------------------------------------------------
        # COVERAGE
        # ---------------------------------------------------------------------

        coverage = decision.get(
            "coverage"
        )

        if coverage is None:

            coverage = decision.get(
                "agent_coverage"
            )

        if coverage is None:

            successful = sum(
                1
                for result in all_results.values()
                if result.get("status") == "success"
            )

            coverage = (
                successful / 6.0
            )

        coverage = self._clip(
            coverage
        )

        # ---------------------------------------------------------------------
        # CLINICAL PREDICTION
        # ---------------------------------------------------------------------

        clinical_prediction = None

        if isinstance(
            clinical_reasoning,
            dict,
        ):

            clinical_prediction = (
                clinical_reasoning.get(
                    "prediction"
                )
            )

        if clinical_prediction is None:

            clinical_prediction = decision.get(
                "clinical_prediction"
            )

        # ---------------------------------------------------------------------
        # CONFLICT SCORE
        # ---------------------------------------------------------------------

        conflict_score = decision.get(
            "conflict_score",
            0.0,
        )

        conflict_score = self._clip(
            conflict_score
        )

        # ---------------------------------------------------------------------
        # AGENT SUMMARY
        # ---------------------------------------------------------------------

        agent_summary = {}

        for name, result in all_results.items():

            agent_summary[name] = {
                "status": result.get(
                    "status"
                ),
                "prediction": result.get(
                    "prediction"
                ),
                "confidence": result.get(
                    "confidence"
                ),
                "uncertainty": result.get(
                    "uncertainty"
                ),
                "trust": result.get(
                    "trust"
                ),
                "quality": result.get(
                    "quality"
                ),
                "task_type": result.get(
                    "task_type"
                ),
            }

        # ---------------------------------------------------------------------
        # FINAL OBJECT
        # ---------------------------------------------------------------------

        return {
            "status": "success",

            "decision": decision_label,

            "confidence": confidence,

            "risk_score": risk_score,

            "coverage": coverage,

            "successful_agents": decision.get(
                "successful_agents",
                sum(
                    1
                    for result in all_results.values()
                    if result.get("status") == "success"
                ),
            ),

            "total_agents": decision.get(
                "num_agents",
                6,
            ),

            "mean_confidence": decision.get(
                "mean_confidence",
                decision.get(
                    "average_confidence",
                    0.0,
                ),
            ),

            "mean_trust": decision.get(
                "mean_trust",
                decision.get(
                    "average_trust",
                    0.0,
                ),
            ),

            "mean_uncertainty": decision.get(
                "mean_uncertainty",
                decision.get(
                    "average_uncertainty",
                    0.0,
                ),
            ),

            "conflict_score": conflict_score,

            "num_conflicts": len(
                conflicts
            ),

            "clinical_prediction": clinical_prediction,

            "clinical_reasoning": clinical_reasoning,

            "agent_summary": agent_summary,

            "weighted_evidence": fusion.get(
                "weighted_evidence",
                fusion.get(
                    "evidence",
                    [],
                ),
            ),

            "fusion": fusion,

            "conflicts": conflicts,

            "decision_engine": decision,

            "explanation": decision.get(
                "explanation"
            ),

            "warning": decision.get(
                "warning"
            ),
        }

    # =========================================================================
    # NOT RUN RESULT
    # =========================================================================

    def _not_run_result(
        self,
        agent_name: str,
        reason: str,
    ) -> Dict[str, Any]:

        return {
            "agent_id": agent_name,
            "agent": agent_name,
            "task_type": agent_name,
            "status": "not_run",
            "prediction": None,
            "probability": None,
            "confidence": 0.0,
            "uncertainty": 1.0,
            "quality": 0.0,
            "trust": 0.0,
            "latency_ms": 0.0,
            "missing_data_ratio": 1.0,
            "error": reason,
        }

    # =========================================================================
    # CLINICAL CONTEXT
    # =========================================================================

    def _build_clinical_context(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        context = {}

        for agent_name, result in results.items():

            if result.get(
                "status"
            ) != "success":
                continue

            context[agent_name] = {
                "prediction": result.get(
                    "prediction"
                ),
                "probability": result.get(
                    "probability"
                ),
                "confidence": result.get(
                    "confidence"
                ),
                "trust": result.get(
                    "trust"
                ),
                "uncertainty": result.get(
                    "uncertainty"
                ),
                "quality": result.get(
                    "quality"
                ),
            }

        return context

    # =========================================================================
    # SAFE FLOAT
    # =========================================================================

    def _safe_float(
        self,
        value,
        default=0.0,
    ):

        try:
            return float(value)

        except (
            ValueError,
            TypeError,
        ):

            return default

    # =========================================================================
    # CLIP
    # =========================================================================

    def _clip(
        self,
        value,
    ):

        value = self._safe_float(
            value,
            default=0.0,
        )

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self):

        agents = {
            "fatty_liver":
                self.fatty_agent is not None,

            "fibrosis":
                self.fibrosis_agent is not None,

            "cirrhosis":
                self.cirrhosis_agent is not None,

            "tumor_classification":
                self.tumor_agent is not None,

            "liver_segmentation":
                self.segmentation_agent is not None,

            "clinical_reasoning":
                self.clinical_reasoning_agent is not None,
        }

        coordinators = {
            "trust_manager":
                self.trust_manager is not None,

            "adaptive_fusion":
                self.adaptive_fusion is not None,

            "conflict_detector":
                self.conflict_detector is not None,

            "decision_engine":
                self.decision_engine is not None,
        }

        initialized_agents = sum(
            agents.values()
        )

        initialized_coordinators = sum(
            coordinators.values()
        )

        if initialized_agents == 6:
            status = "ok"

        elif initialized_agents > 0:
            status = "partial"

        else:
            status = "error"

        return {
            "status": status,

            "agents": agents,

            "coordinators": coordinators,

            "initialized_agents":
                initialized_agents,

            "total_agents": 6,

            "initialized_coordinators":
                initialized_coordinators,

            "total_coordinators": 4,

            "initialization_errors":
                self.initialization_errors,
        }

    # =========================================================================
    # GETTERS
    # =========================================================================

    def get_last_results(self):
        return self.last_results

    def get_last_assessment(self):
        return self.last_final_decision

    def get_execution_log(self):
        return self.execution_log
