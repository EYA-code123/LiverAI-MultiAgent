# ================================================================
# LIVER AI MULTI-AGENT
# FILE: orchestrator/liver_orchestrator.py
# VERSION: FINAL COORDINATION VERSION
# ================================================================

from typing import Dict, Any, Optional
from datetime import datetime
import time
import traceback

import joblib


# ================================================================
# AGENTS
# ================================================================

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
    print(
        "WARNING: TumorClassificationAgent import failed:",
        repr(e)
    )


try:
    from agents.liver_segmentation_agent import LiverSegmentationAgent
except Exception as e:
    LiverSegmentationAgent = None
    print(
        "WARNING: LiverSegmentationAgent import failed:",
        repr(e)
    )


try:
    from agents.clinical_reasoning_agent import ClinicalReasoningAgent
except Exception as e:
    ClinicalReasoningAgent = None
    print(
        "WARNING: ClinicalReasoningAgent import failed:",
        repr(e)
    )


# ================================================================
# ORCHESTRATION / COORDINATION COMPONENTS
# ================================================================

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


# ================================================================
# MODEL PATHS
# ================================================================

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


# ================================================================
# ORCHESTRATOR
# ================================================================

class LiverAIOrchestrator:
    """
    Main orchestrator for the six Liver AI agents.

    Agents:
        1. Fatty Liver
        2. Fibrosis
        3. Cirrhosis
        4. Tumor Classification
        5. Liver Segmentation
        6. Clinical Reasoning

    Pipeline:

        Input
          |
          v
        Six specialized agents
          |
          v
        Result normalization
          |
          v
        Trust estimation
          |
          v
        Adaptive fusion
          |
          v
        Conflict detection
          |
          v
        Decision engine / fallback decision
          |
          v
        Final structured output

    IMPORTANT:
        The six agents do NOT solve the same classification task.
        Therefore their labels must NOT be treated as a single
        common class label.

        Example:
            fatty_liver = 2
            fibrosis = 0
            cirrhosis = 2
            tumor = Healthy
            segmentation = mask/statistics
            clinical_reasoning = 1

        These outputs represent different clinical domains.
        The final decision therefore summarizes the evidence
        by domain rather than pretending all predictions belong
        to one common label space.
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

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

        # Automatic initialization
        auto_initialize=True,
        device=None,
    ):

        # --------------------------------------------------------
        # DEVICE
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # INITIALIZATION ERRORS
        # --------------------------------------------------------

        self.initialization_errors = {}

        # --------------------------------------------------------
        # EXPLICIT AGENTS
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # COORDINATORS
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # AUTOMATIC AGENT INITIALIZATION
        # --------------------------------------------------------

        if auto_initialize:
            self._initialize_agents()

        # --------------------------------------------------------
        # AGENT REGISTRY
        # --------------------------------------------------------

        self.agent_registry = {
            "fatty_liver": self.fatty_agent,
            "fibrosis": self.fibrosis_agent,
            "cirrhosis": self.cirrhosis_agent,
            "tumor_classification": self.tumor_agent,
            "liver_segmentation": self.segmentation_agent,
            "clinical_reasoning": self.clinical_reasoning_agent,
        }

    # ============================================================
    # INITIALIZE ALL SIX AGENTS
    # ============================================================

    def _initialize_agents(self):

        print("=" * 70)
        print("LIVER AI AGENTS INITIALIZATION")
        print("=" * 70)

        # --------------------------------------------------------
        # 1. FATTY LIVER
        # --------------------------------------------------------

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

                print(
                    "✓ Fatty Liver Agent initialized"
                )

            except Exception as e:

                self.initialization_errors[
                    "fatty_liver"
                ] = repr(e)

                print(
                    "❌ Fatty Liver Agent:",
                    repr(e)
                )

        else:

            print(
                "✓ Fatty Liver Agent supplied externally"
            )

        # --------------------------------------------------------
        # 2. FIBROSIS
        # --------------------------------------------------------

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

                print(
                    "✓ Fibrosis Agent initialized"
                )

            except Exception as e:

                self.initialization_errors[
                    "fibrosis"
                ] = repr(e)

                print(
                    "❌ Fibrosis Agent:",
                    repr(e)
                )

        else:

            print(
                "✓ Fibrosis Agent supplied externally"
            )

        # --------------------------------------------------------
        # 3. CIRRHOSIS
        # --------------------------------------------------------

        if self.cirrhosis_agent is None:

            try:

                if CirrhosisAgent is None:
                    raise ImportError(
                        "CirrhosisAgent could not be imported."
                    )

                self.cirrhosis_agent = CirrhosisAgent(
                    CIRRHOSIS_MODEL_PATH
                )

                print(
                    "✓ Cirrhosis Agent initialized"
                )

            except Exception as e:

                self.initialization_errors[
                    "cirrhosis"
                ] = repr(e)

                print(
                    "❌ Cirrhosis Agent:",
                    repr(e)
                )

        else:

            print(
                "✓ Cirrhosis Agent supplied externally"
            )

        # --------------------------------------------------------
        # 4. TUMOR CLASSIFICATION
        # --------------------------------------------------------

        if self.tumor_agent is None:

            try:

                if TumorClassificationAgent is None:
                    raise ImportError(
                        "TumorClassificationAgent "
                        "could not be imported."
                    )

                self.tumor_agent = (
                    TumorClassificationAgent(
                        TUMOR_MODEL_PATH
                    )
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
                "✓ Tumor Classification Agent "
                "supplied externally"
            )

        # --------------------------------------------------------
        # 5. LIVER SEGMENTATION
        # --------------------------------------------------------

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
                        target_size=(
                            128,
                            128,
                            64
                        ),
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
                "✓ Liver Segmentation Agent "
                "supplied externally"
            )

        # --------------------------------------------------------
        # 6. CLINICAL REASONING
        # --------------------------------------------------------

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
                "✓ Clinical Reasoning Agent "
                "supplied externally"
            )

        # --------------------------------------------------------
        # SUMMARY
        # --------------------------------------------------------

        print("=" * 70)

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

            if self.initialization_errors:

                print(
                    "\nInitialization errors:"
                )

                for name, error in (
                    self.initialization_errors.items()
                ):

                    print(
                        f"  - {name}: {error}"
                    )

        print("=" * 70)

    # ============================================================
    # MAIN RUN METHOD
    # ============================================================

    def run(
        self,
        patient_id: str = "UNKNOWN",
        patient_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        start_time = time.perf_counter()

        if patient_data is None:
            patient_data = {}

        # --------------------------------------------------------
        # SPECIALIZED AGENTS
        # --------------------------------------------------------

        specialized_results = (
            self.run_specialized_agents(
                patient_data
            )
        )

        # --------------------------------------------------------
        # CLINICAL REASONING
        # --------------------------------------------------------

        clinical_result = (
            self.run_clinical_reasoning(
                patient_data,
                specialized_results,
            )
        )

        # --------------------------------------------------------
        # COMBINE RESULTS
        # --------------------------------------------------------

        all_results = dict(
            specialized_results
        )

        all_results[
            "clinical_reasoning"
        ] = clinical_result

        # --------------------------------------------------------
        # ADAPTIVE FUSION
        # --------------------------------------------------------

        fusion = self._run_adaptive_fusion(
            all_results
        )

        # --------------------------------------------------------
        # CONFLICT DETECTION
        # --------------------------------------------------------

        conflicts = self._run_conflict_detection(
            all_results
        )

        # --------------------------------------------------------
        # DECISION ENGINE
        # --------------------------------------------------------

        decision = self._run_decision_engine(
            all_results,
            conflicts,
            clinical_result,
            fusion=fusion,
        )

        # --------------------------------------------------------
        # FINAL DECISION
        # --------------------------------------------------------

        final_decision = (
            self._build_final_decision(
                all_results=all_results,
                fusion=fusion,
                conflicts=conflicts,
                decision=decision,
            )
        )

        # --------------------------------------------------------
        # STATUS
        # --------------------------------------------------------

        successful_agents = sum(
            1
            for result in all_results.values()
            if isinstance(result, dict)
            and result.get("status") == "success"
        )

        failed_agents = [
            name
            for name, result in all_results.items()
            if isinstance(result, dict)
            and result.get("status") == "error"
        ]

        not_run_agents = [
            name
            for name, result in all_results.items()
            if isinstance(result, dict)
            and result.get("status") == "not_run"
        ]

        total_agents = 6

        if successful_agents == 0:

            overall_status = "failed"

        elif successful_agents == total_agents:

            overall_status = "success"

        else:

            overall_status = "partial"

        # --------------------------------------------------------
        # LATENCY
        # --------------------------------------------------------

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        # --------------------------------------------------------
        # FINAL OUTPUT
        # --------------------------------------------------------

        return {
            "status": overall_status,

            "patient_id": patient_id,

            "timestamp": datetime.now().isoformat(),

            # Explicit specialized outputs
            "specialized_results": specialized_results,

            # Clinical reasoning explicitly exposed
            "clinical_reasoning": clinical_result,

            # All six results
            "agent_results": all_results,

            # Fusion
            "fusion": fusion,

            # Conflicts
            "conflicts": conflicts,

            # Decision engine result
            "decision": decision,

            # Final structured decision
            "final_decision": final_decision,

            # Clinical context
            "clinical_context": (
                self._build_clinical_context(
                    all_results
                )
            ),

            # Coordination
            "coordination": {
                "total_agents": total_agents,

                "successful_agents": (
                    successful_agents
                ),

                "failed_agents": failed_agents,

                "not_run_agents": not_run_agents,

                "coverage": round(
                    successful_agents / total_agents,
                    4,
                ),

                "latency_ms": round(
                    latency_ms,
                    3,
                ),
            },
        }

    # ============================================================
    # ALIASES
    # ============================================================

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

    # ============================================================
    # SPECIALIZED AGENTS
    # ============================================================

    def run_specialized_agents(
        self,
        patient_data: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:

        results = {}

        # --------------------------------------------------------
        # 1. FATTY LIVER
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # 2. FIBROSIS
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # 3. CIRRHOSIS
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # 4. TUMOR CLASSIFICATION
        # --------------------------------------------------------

        tumor_input = patient_data.get(
            "tumor"
        )

        if tumor_input is None:

            tumor_input = patient_data.get(
                "tumor_classification"
            )

        if tumor_input is not None:

            results[
                "tumor_classification"
            ] = self._execute_agent(
                agent_name="tumor_classification",
                agent=self.tumor_agent,
                input_data=tumor_input,
            )

        else:

            results[
                "tumor_classification"
            ] = self._not_run_result(
                "tumor_classification",
                "No tumor image provided.",
            )

        # --------------------------------------------------------
        # 5. LIVER SEGMENTATION
        # --------------------------------------------------------

        segmentation_input = patient_data.get(
            "segmentation"
        )

        if segmentation_input is None:

            segmentation_input = patient_data.get(
                "liver_segmentation"
            )

        if segmentation_input is not None:

            results[
                "liver_segmentation"
            ] = self._execute_agent(
                agent_name="liver_segmentation",
                agent=self.segmentation_agent,
                input_data=segmentation_input,
            )

        else:

            results[
                "liver_segmentation"
            ] = self._not_run_result(
                "liver_segmentation",
                "No liver volume provided.",
            )

        return results

    # ============================================================
    # CLINICAL REASONING
    # ============================================================

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

        # --------------------------------------------------------
        # Clinical model expects BUPA-style six features.
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # EXECUTE
        # --------------------------------------------------------

        result = self._execute_agent(
            agent_name="clinical_reasoning",
            agent=self.clinical_reasoning_agent,
            input_data=clinical_input,
        )

        # --------------------------------------------------------
        # ATTACH SPECIALIZED EVIDENCE
        # --------------------------------------------------------

        result["specialized_evidence"] = {}

        for name, agent_result in (
            specialized_results.items()
        ):

            if (
                isinstance(agent_result, dict)
                and agent_result.get("status")
                == "success"
            ):

                result[
                    "specialized_evidence"
                ][name] = {

                    "prediction": (
                        agent_result.get(
                            "prediction"
                        )
                    ),

                    "confidence": (
                        agent_result.get(
                            "confidence"
                        )
                    ),

                    "trust": (
                        agent_result.get(
                            "trust"
                        )
                    ),

                    "probability": (
                        agent_result.get(
                            "probability"
                        )
                    ),

                    "uncertainty": (
                        agent_result.get(
                            "uncertainty"
                        )
                    ),
                }

        return result

    # ============================================================
    # EXECUTE AGENT
    # ============================================================

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

            # ----------------------------------------------------
            # FIND PREDICTION METHOD
            # ----------------------------------------------------

            if hasattr(agent, "predict"):

                raw_result = agent.predict(
                    input_data
                )

            elif hasattr(agent, "analyze"):

                raw_result = agent.analyze(
                    input_data
                )

            elif hasattr(agent, "run"):

                raw_result = agent.run(
                    input_data
                )

            else:

                raise AttributeError(
                    f"{type(agent).__name__} "
                    "has no predict(), analyze(), "
                    "or run() method."
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

    # ============================================================
    # NORMALIZE RESULT
    # ============================================================

    def _normalize_result(
        self,
        agent_name: str,
        result: Any,
        latency_ms: float = 0.0,
    ) -> Dict[str, Any]:

        # --------------------------------------------------------
        # NONE
        # --------------------------------------------------------

        if result is None:

            return {
                "agent_id": agent_name,
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

        # --------------------------------------------------------
        # DICT
        # --------------------------------------------------------

        if isinstance(result, dict):

            normalized = dict(result)

            normalized.setdefault(
                "agent_id",
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

            # ----------------------------------------------------
            # CONFIDENCE
            # ----------------------------------------------------

            confidence = (
                self._extract_confidence(
                    normalized
                )
            )

            normalized[
                "confidence"
            ] = confidence

            # ----------------------------------------------------
            # UNCERTAINTY
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # QUALITY
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # TRUST
            # ----------------------------------------------------

            normalized[
                "trust"
            ] = self._compute_trust(
                agent_name,
                normalized,
            )

            # ----------------------------------------------------
            # MODALITY
            # ----------------------------------------------------

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
                    ] = "image_2d"

                elif agent_name == (
                    "liver_segmentation"
                ):

                    normalized[
                        "modality"
                    ] = "volume_3d"

                else:

                    normalized[
                        "modality"
                    ] = "unknown"

            return normalized

        # --------------------------------------------------------
        # NON-DICT
        # --------------------------------------------------------

        confidence = 0.5

        normalized = {
            "agent_id": agent_name,
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

    # ============================================================
    # CONFIDENCE EXTRACTION
    # ============================================================

    def _extract_confidence(
        self,
        result: Dict[str, Any],
    ) -> float:

        # --------------------------------------------------------
        # EXPLICIT CONFIDENCE
        # --------------------------------------------------------

        confidence = result.get(
            "confidence"
        )

        if confidence is not None:

            value = self._safe_float(
                confidence,
                default=None,
            )

            if value is not None:

                return self._clip(
                    value
                )

        # --------------------------------------------------------
        # PROBABILITY
        # --------------------------------------------------------

        probability = result.get(
            "probability"
        )

        if probability is not None:

            value = (
                self._probability_confidence(
                    probability
                )
            )

            if value is not None:

                return self._clip(
                    value
                )

        # --------------------------------------------------------
        # CLASS PROBABILITIES
        # --------------------------------------------------------

        class_probabilities = (
            result.get(
                "class_probabilities"
            )
        )

        if class_probabilities is not None:

            value = (
                self._probability_confidence(
                    class_probabilities
                )
            )

            if value is not None:

                return self._clip(
                    value
                )

        # --------------------------------------------------------
        # PROBABILITIES
        # --------------------------------------------------------

        probabilities = result.get(
            "probabilities"
        )

        if probabilities is not None:

            value = (
                self._probability_confidence(
                    probabilities
                )
            )

            if value is not None:

                return self._clip(
                    value
                )

        return 0.5

    # ============================================================
    # PROBABILITY CONFIDENCE
    # ============================================================

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

                for value in (
                    probabilities.values()
                ):

                    try:

                        values.append(
                            float(value)
                        )

                    except (
                        ValueError,
                        TypeError,
                    ):

                        pass

                if values:

                    return max(values)

                return None

            if isinstance(
                probabilities,
                (list, tuple),
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

    # ============================================================
    # TRUST
    # ============================================================

    def _compute_trust(
        self,
        agent_name: str,
        result: Dict[str, Any],
    ) -> float:

        # --------------------------------------------------------
        # NO TRUST MANAGER
        # --------------------------------------------------------

        if self.trust_manager is None:

            # Reasonable fallback instead of zero.
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

            fallback = (
                0.45 * confidence
                + 0.35 * quality
                + 0.20 * (1.0 - uncertainty)
            )

            fallback *= (
                1.0 - 0.5 * missing_ratio
            )

            return self._clip(
                fallback
            )

        # --------------------------------------------------------
        # INPUTS
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # TRUST MANAGER
        # --------------------------------------------------------

        try:

            trust = (
                self.trust_manager.compute_trust(
                    agent_id=agent_name,
                    confidence=confidence,
                    quality=quality,
                    uncertainty=uncertainty,
                    missing_data_ratio=missing_ratio,
                    agreement=0.5,
                    stability=0.5,
                    utility=0.5,
                    modality_available=True,
                )
            )

            return self._clip(
                trust
            )

        except TypeError:

            try:

                trust = (
                    self.trust_manager.compute_trust(
                        agent_name,
                        confidence,
                        quality,
                        uncertainty,
                        missing_ratio,
                    )
                )

                return self._clip(
                    trust
                )

            except Exception:

                return self._fallback_trust(
                    confidence,
                    quality,
                    uncertainty,
                    missing_ratio,
                )

        except Exception:

            return self._fallback_trust(
                confidence,
                quality,
                uncertainty,
                missing_ratio,
            )

    # ============================================================
    # FALLBACK TRUST
    # ============================================================

    def _fallback_trust(
        self,
        confidence,
        quality,
        uncertainty,
        missing_ratio,
    ):

        value = (
            0.45 * confidence
            + 0.35 * quality
            + 0.20 * (1.0 - uncertainty)
        )

        value *= (
            1.0 - 0.5 * missing_ratio
        )

        return self._clip(
            value
        )

    # ============================================================
    # ADAPTIVE FUSION
    # ============================================================

    def _run_adaptive_fusion(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        if self.adaptive_fusion is None:

            return self._fallback_fusion(
                results
            )

        valid_results = [
            result
            for result in results.values()
            if isinstance(result, dict)
            and result.get("status") == "success"
        ]

        if not valid_results:

            return {
                "status": "no_valid_results",
                "evidence": [],
                "weights": {},
                "task_groups": {},
                "same_task_fusion": {},
                "coverage": 0.0,
            }

        try:

            fusion = (
                self.adaptive_fusion.fuse(
                    valid_results
                )
            )

            if isinstance(
                fusion,
                dict,
            ):

                # Make sure useful metadata exists.
                fusion.setdefault(
                    "status",
                    "success",
                )

                fusion.setdefault(
                    "coverage",
                    len(valid_results) / 6.0,
                )

                return fusion

            return {
                "status": "success",
                "result": fusion,
                "evidence": valid_results,
                "coverage": (
                    len(valid_results) / 6.0
                ),
            }

        except Exception as e:

            fallback = self._fallback_fusion(
                results
            )

            fallback[
                "adaptive_fusion_error"
            ] = repr(e)

            return fallback

    # ============================================================
    # FALLBACK FUSION
    # ============================================================

    def _fallback_fusion(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        valid_results = {
            name: result
            for name, result in results.items()
            if isinstance(result, dict)
            and result.get("status") == "success"
        }

        evidence = []

        weights = {}

        for name, result in valid_results.items():

            confidence = self._clip(
                result.get(
                    "confidence",
                    0.0,
                )
            )

            trust = self._clip(
                result.get(
                    "trust",
                    0.0,
                )
            )

            quality = self._clip(
                result.get(
                    "quality",
                    1.0,
                )
            )

            weight = (
                0.50 * confidence
                + 0.30 * trust
                + 0.20 * quality
            )

            weights[name] = round(
                weight,
                6,
            )

            evidence.append({
                "agent": name,
                "prediction": result.get(
                    "prediction"
                ),
                "confidence": confidence,
                "trust": trust,
                "quality": quality,
                "weight": round(
                    weight,
                    6,
                ),
            })

        return {
            "status": (
                "success"
                if valid_results
                else "no_valid_results"
            ),

            "method": (
                "fallback_heterogeneous_evidence"
            ),

            "evidence": evidence,

            "weights": weights,

            "task_groups": {},

            "same_task_fusion": {},

            "coverage": (
                len(valid_results) / 6.0
            ),

            "successful_agents": (
                len(valid_results)
            ),

            "total_agents": 6,
        }

    # ============================================================
    # CONFLICT DETECTION
    # ============================================================

    def _run_conflict_detection(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> list:

        if self.conflict_detector is None:

            return []

        valid_results = [
            result
            for result in results.values()
            if isinstance(result, dict)
            and result.get("status") == "success"
        ]

        if len(valid_results) < 2:

            return []

        # --------------------------------------------------------
        # AgentResult conversion
        # --------------------------------------------------------

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
                                "agent_id"
                            ),

                            task_type=result.get(
                                "task_type"
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

                            missing_data_ratio=(
                                result.get(
                                    "missing_data_ratio",
                                    0.0,
                                )
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

            if isinstance(
                conflicts,
                list,
            ):

                return conflicts

            return [
                conflicts
            ]

        except Exception as e:

            return [
                {
                    "type": "conflict_detection_error",
                    "error": repr(e),
                }
            ]

    # ============================================================
    # DECISION ENGINE
    # ============================================================

    def _run_decision_engine(
        self,
        results: Dict[str, Dict[str, Any]],
        conflicts,
        clinical_result: Dict[str, Any],
        fusion: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        # --------------------------------------------------------
        # NO DECISION ENGINE
        # --------------------------------------------------------

        if self.decision_engine is None:

            return {
                "status": "fallback",
                "source": "orchestrator",
                "prediction": None,
                "decision_level": "UNCERTAIN",
                "confidence": 0.0,
                "uncertainty": 1.0,
                "trust": 0.0,
                "quality": 0.0,
                "coverage": self._calculate_coverage(
                    results
                ),
                "reason": (
                    "DecisionEngine unavailable."
                ),
            }

        all_results = list(
            results.values()
        )

        # --------------------------------------------------------
        # TRY DECISION ENGINE
        # --------------------------------------------------------

        try:

            decision = (
                self.decision_engine.decide(
                    all_results,
                    conflicts,
                    clinical_result,
                )
            )

            # ----------------------------------------------------
            # DICT
            # ----------------------------------------------------

            if isinstance(
                decision,
                dict,
            ):

                # Do not allow a missing/null prediction
                # to make the complete decision unusable.
                if (
                    decision.get(
                        "prediction"
                    ) is None
                    and decision.get(
                        "result"
                    ) is None
                ):

                    decision[
                        "status"
                    ] = "fallback_required"

                else:

                    decision.setdefault(
                        "status",
                        "completed",
                    )

                decision.setdefault(
                    "coverage",
                    self._calculate_coverage(
                        results
                    ),
                )

                return decision

            # ----------------------------------------------------
            # NON-DICT
            # ----------------------------------------------------

            if decision is not None:

                return {
                    "status": "completed",
                    "source": "decision_engine",
                    "result": decision,
                    "prediction": None,
                    "coverage": (
                        self._calculate_coverage(
                            results
                        )
                    ),
                }

            # ----------------------------------------------------
            # NONE
            # ----------------------------------------------------

            return {
                "status": "fallback_required",
                "source": "decision_engine",
                "prediction": None,
                "coverage": (
                    self._calculate_coverage(
                        results
                    )
                ),
                "reason": (
                    "DecisionEngine returned None."
                ),
            }

        except Exception as e:

            return {
                "status": "fallback_required",

                "source": "decision_engine",

                "prediction": None,

                "decision_level": "UNCERTAIN",

                "confidence": 0.0,

                "uncertainty": 1.0,

                "trust": 0.0,

                "quality": 0.0,

                "coverage": (
                    self._calculate_coverage(
                        results
                    )
                ),

                "error": repr(e),

                "reason": (
                    "DecisionEngine execution failed."
                ),
            }

    # ============================================================
    # FINAL DECISION
    # ============================================================

    def _build_final_decision(
        self,
        all_results: Dict[str, Dict[str, Any]],
        fusion: Dict[str, Any],
        conflicts,
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:

        coverage = self._calculate_coverage(
            all_results
        )

        successful = {
            name: result
            for name, result
            in all_results.items()
            if isinstance(result, dict)
            and result.get("status") == "success"
        }

        # --------------------------------------------------------
        # DOMAIN FINDINGS
        # --------------------------------------------------------

        findings = {}

        for agent_name, result in successful.items():

            finding = {
                "status": "available",

                "prediction": result.get(
                    "prediction"
                ),

                "confidence": self._clip(
                    result.get(
                        "confidence",
                        0.0,
                    )
                ),

                "uncertainty": self._clip(
                    result.get(
                        "uncertainty",
                        1.0,
                    )
                ),

                "trust": self._clip(
                    result.get(
                        "trust",
                        0.0,
                    )
                ),

                "quality": self._clip(
                    result.get(
                        "quality",
                        1.0,
                    )
                ),

                "modality": result.get(
                    "modality"
                ),
            }

            # ----------------------------------------------------
            # TUMOR DETAILS
            # ----------------------------------------------------

            if agent_name == (
                "tumor_classification"
            ):

                finding[
                    "model_prediction"
                ] = result.get(
                    "prediction"
                )

                finding[
                    "class_probabilities"
                ] = result.get(
                    "class_probabilities"
                )

                finding[
                    "probabilities"
                ] = result.get(
                    "probabilities"
                )

            # ----------------------------------------------------
            # SEGMENTATION DETAILS
            # ----------------------------------------------------

            if agent_name == (
                "liver_segmentation"
            ):

                finding[
                    "liver_voxels"
                ] = result.get(
                    "liver_voxels"
                )

                finding[
                    "total_voxels"
                ] = result.get(
                    "total_voxels"
                )

                finding[
                    "liver_ratio"
                ] = result.get(
                    "liver_ratio"
                )

                finding[
                    "mean_probability"
                ] = result.get(
                    "mean_probability"
                )

                finding[
                    "max_probability"
                ] = result.get(
                    "max_probability"
                )

                finding[
                    "input_shape"
                ] = result.get(
                    "input_shape"
                )

                finding[
                    "output_shape"
                ] = result.get(
                    "output_shape"
                )

            # ----------------------------------------------------
            # CLINICAL REASONING DETAILS
            # ----------------------------------------------------

            if agent_name == (
                "clinical_reasoning"
            ):

                finding[
                    "probabilities"
                ] = result.get(
                    "probabilities"
                )

                finding[
                    "specialized_evidence"
                ] = result.get(
                    "specialized_evidence",
                    {},
                )

            findings[
                agent_name
            ] = finding

        # --------------------------------------------------------
        # HIGH / MEDIUM / LOW CONFIDENCE
        # --------------------------------------------------------

        high_confidence = []
        medium_confidence = []
        low_confidence = []

        for name, finding in findings.items():

            confidence = finding[
                "confidence"
            ]

            if confidence >= 0.80:

                high_confidence.append(
                    name
                )

            elif confidence >= 0.60:

                medium_confidence.append(
                    name
                )

            else:

                low_confidence.append(
                    name
                )

        # --------------------------------------------------------
        # CLINICAL REASONING
        # --------------------------------------------------------

        clinical_result = all_results.get(
            "clinical_reasoning",
            {},
        )

        clinical_prediction = None
        clinical_confidence = 0.0

        if isinstance(
            clinical_result,
            dict,
        ):

            clinical_prediction = (
                clinical_result.get(
                    "prediction"
                )
            )

            clinical_confidence = self._clip(
                clinical_result.get(
                    "confidence",
                    0.0,
                )
            )

        # --------------------------------------------------------
        # DECISION ENGINE OUTPUT
        # --------------------------------------------------------

        engine_prediction = None
        engine_confidence = 0.0

        if isinstance(
            decision,
            dict,
        ):

            engine_prediction = (
                decision.get(
                    "prediction"
                )
            )

            engine_confidence = self._clip(
                decision.get(
                    "confidence",
                    0.0,
                )
            )

        # --------------------------------------------------------
        # GLOBAL CONFIDENCE
        # --------------------------------------------------------

        confidence_values = []

        for result in successful.values():

            value = self._safe_float(
                result.get(
                    "confidence",
                    0.0,
                ),
                default=0.0,
            )

            confidence_values.append(
                self._clip(value)
            )

        if confidence_values:

            mean_confidence = (
                sum(confidence_values)
                / len(confidence_values)
            )

        else:

            mean_confidence = 0.0

        # --------------------------------------------------------
        # CONFLICT STATUS
        # --------------------------------------------------------

        has_conflicts = (
            isinstance(conflicts, list)
            and len(conflicts) > 0
        )

        # --------------------------------------------------------
        # OVERALL EVIDENCE STATUS
        # --------------------------------------------------------

        if coverage >= 1.0:

            evidence_status = (
                "complete_coverage"
            )

        elif coverage > 0.0:

            evidence_status = (
                "partial_coverage"
            )

        else:

            evidence_status = (
                "no_valid_evidence"
            )

        # --------------------------------------------------------
        # OVERALL DECISION LEVEL
        # --------------------------------------------------------

        if coverage == 0.0:

            decision_level = "INSUFFICIENT_DATA"

        elif has_conflicts:

            decision_level = "CONFLICTING_EVIDENCE"

        elif mean_confidence < 0.50:

            decision_level = "LOW_CONFIDENCE"

        elif mean_confidence < 0.70:

            decision_level = "MODERATE_CONFIDENCE"

        else:

            decision_level = "HIGHER_CONFIDENCE"

        # --------------------------------------------------------
        # SUMMARY
        # --------------------------------------------------------

        summary = (
            "Multi-agent liver assessment completed. "
            "Results are reported by clinical domain. "
            "The heterogeneous agent outputs should not "
            "be interpreted as a single common diagnostic label."
        )

        # --------------------------------------------------------
        # RECOMMENDATION / NEXT STEP
        # --------------------------------------------------------

        if coverage < 1.0:

            next_step = (
                "Complete the missing modalities or inputs "
                "before considering the assessment complete."
            )

        elif has_conflicts:

            next_step = (
                "Review conflicting agent evidence "
                "before relying on the combined assessment."
            )

        elif mean_confidence < 0.60:

            next_step = (
                "The model evidence is relatively uncertain; "
                "additional validated clinical information "
                "should be considered."
            )

        else:

            next_step = (
                "Use the domain-specific model outputs as "
                "decision-support evidence and review them "
                "with appropriate clinical context."
            )

        # --------------------------------------------------------
        # FINAL STRUCTURED RESULT
        # --------------------------------------------------------

        return {

            "status": "completed",

            "decision_level": decision_level,

            "evidence_status": evidence_status,

            "coverage": round(
                coverage,
                4,
            ),

            "successful_agents": len(
                successful
            ),

            "total_agents": 6,

            "mean_confidence": round(
                mean_confidence,
                4,
            ),

            "summary": summary,

            "next_step": next_step,

            # Domain-specific findings
            "findings": findings,

            "confidence_groups": {
                "high": high_confidence,
                "medium": medium_confidence,
                "low": low_confidence,
            },

            # Clinical reasoning
            "clinical_reasoning": {
                "prediction": (
                    clinical_prediction
                ),
                "confidence": (
                    clinical_confidence
                ),
            },

            # Decision engine output
            "decision_engine": {
                "status": (
                    decision.get(
                        "status"
                    )
                    if isinstance(
                        decision,
                        dict,
                    )
                    else None
                ),

                "prediction": (
                    engine_prediction
                ),

                "confidence": (
                    engine_confidence
                ),
            },

            # Fusion metadata
            "fusion": {
                "status": (
                    fusion.get(
                        "status"
                    )
                    if isinstance(
                        fusion,
                        dict,
                    )
                    else None
                ),

                "method": (
                    fusion.get(
                        "fusion_method"
                    )
                    or fusion.get(
                        "method"
                    )
                    if isinstance(
                        fusion,
                        dict,
                    )
                    else None
                ),

                "coverage": (
                    fusion.get(
                        "coverage"
                    )
                    if isinstance(
                        fusion,
                        dict,
                    )
                    else None
                ),

                "successful_agents": (
                    fusion.get(
                        "successful_agents"
                    )
                    if isinstance(
                        fusion,
                        dict,
                    )
                    else None
                ),
            },

            # Conflict metadata
            "conflicts": {
                "count": (
                    len(conflicts)
                    if isinstance(
                        conflicts,
                        list,
                    )
                    else 0
                ),

                "present": has_conflicts,
            },
        }

    # ============================================================
    # CALCULATE COVERAGE
    # ============================================================

    def _calculate_coverage(
        self,
        results,
    ) -> float:

        if not isinstance(
            results,
            dict,
        ):

            return 0.0

        successful = sum(
            1
            for result in results.values()
            if isinstance(result, dict)
            and result.get("status") == "success"
        )

        return round(
            successful / 6.0,
            4,
        )

    # ============================================================
    # NOT RUN RESULT
    # ============================================================

    def _not_run_result(
        self,
        agent_name: str,
        reason: str,
    ) -> Dict[str, Any]:

        return {
            "agent_id": agent_name,

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

    # ============================================================
    # FIND RESULT
    # ============================================================

    def _find_result(
        self,
        results,
        agent_name,
    ):

        if isinstance(
            results,
            dict,
        ):

            return results.get(
                agent_name
            )

        return None

    # ============================================================
    # CLINICAL CONTEXT
    # ============================================================

    def _build_clinical_context(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        context = {}

        for agent_name, result in (
            results.items()
        ):

            if (
                not isinstance(
                    result,
                    dict,
                )
                or result.get(
                    "status"
                ) != "success"
            ):

                continue

            context[
                agent_name
            ] = {

                "prediction": (
                    result.get(
                        "prediction"
                    )
                ),

                "probability": (
                    result.get(
                        "probability"
                    )
                ),

                "confidence": (
                    result.get(
                        "confidence"
                    )
                ),

                "uncertainty": (
                    result.get(
                        "uncertainty"
                    )
                ),

                "trust": (
                    result.get(
                        "trust"
                    )
                ),

                "quality": (
                    result.get(
                        "quality"
                    )
                ),

                "modality": (
                    result.get(
                        "modality"
                    )
                ),
            }

        return context

    # ============================================================
    # SAFE FLOAT
    # ============================================================

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

    # ============================================================
    # CLIP
    # ============================================================

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

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def health_check(self):

        agents = {

            "fatty_liver": (
                self.fatty_agent is not None
            ),

            "fibrosis": (
                self.fibrosis_agent is not None
            ),

            "cirrhosis": (
                self.cirrhosis_agent is not None
            ),

            "tumor_classification": (
                self.tumor_agent is not None
            ),

            "liver_segmentation": (
                self.segmentation_agent is not None
            ),

            "clinical_reasoning": (
                self.clinical_reasoning_agent
                is not None
            ),
        }

        coordinators = {

            "trust_manager": (
                self.trust_manager is not None
            ),

            "adaptive_fusion": (
                self.adaptive_fusion is not None
            ),

            "conflict_detector": (
                self.conflict_detector is not None
            ),

            "decision_engine": (
                self.decision_engine is not None
            ),
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

            "initialized_agents": (
                initialized_agents
            ),

            "total_agents": 6,

            "initialized_coordinators": (
                initialized_coordinators
            ),

            "total_coordinators": 4,

            "device": self.device,

            "initialization_errors": (
                self.initialization_errors
            ),
        }


# ================================================================
# COMPATIBILITY ALIASES
# ================================================================

LiverOrchestrator = LiverAIOrchestrator


__all__ = [
    "LiverAIOrchestrator",
    "LiverOrchestrator",
]
