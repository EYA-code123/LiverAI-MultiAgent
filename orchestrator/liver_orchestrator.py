# ================================================================
# LiverAI Multi-Agent System
# Central Liver AI Orchestrator
# ================================================================

from typing import Dict, Any, Optional
from datetime import datetime
import os
import time
import traceback

import joblib


# ================================================================
# OPTIONAL COORDINATION COMPONENTS
# ================================================================

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


# ================================================================
# AGENTS
# ================================================================

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


# ================================================================
# MODEL PATHS
# ================================================================

FATTY_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "Fatty_Liver_Dataset/models/FattyLiver_LightGBM.pkl"
)

FIBROSIS_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "Fibrosis Agent/XGBoost_model/xgboost_nafld.pkl"
)

CIRRHOSIS_MODEL_PATH = (
    "/content/drive/MyDrive/"
    ".Cirrhosis Agent/XGBoost_model/"
    "XGBoost_Cirrhosis_fixed.joblib"
)

TUMOR_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "models/tumor/efficientnet_b0_best.pth"
)

SEGMENTATION_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "Liver Segmentation Agent/models/"
    "SegResNet3D_Liver_best.pth"
)

CLINICAL_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "Clinical Reasoning Agent/tabtransformer_bupa"
)


# ================================================================
# MAIN ORCHESTRATOR
# ================================================================

class LiverAIOrchestrator:
    """
    Central orchestrator for the LiverAI multi-agent system.

    Agents:
        1. Fatty Liver
        2. Fibrosis
        3. Cirrhosis
        4. Tumor Classification
        5. Liver Segmentation
        6. Clinical Reasoning

    Architecture:

        Patient Input
             |
             v
        LiverAIOrchestrator
             |
        +----+----+----+----+----+
        |    |    |    |    |    |
       FL   FIB   CIR  TUM  SEG  CR
        |    |    |    |    |    |
        +----+----+----+----+----+
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
          Unified Liver Report
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

        # Automatically load models
        auto_initialize=True,
        device=None,
    ):

        # --------------------------------------------------------
        # Device
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
        # Initialization errors
        # --------------------------------------------------------

        self.initialization_errors = {}

        # --------------------------------------------------------
        # User-supplied agents
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
        # Automatically initialize missing agents
        # --------------------------------------------------------

        if auto_initialize:
            self._initialize_agents()

        # --------------------------------------------------------
        # Coordinator modules
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
        # Agent registry
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
    # AUTOMATIC AGENT INITIALIZATION
    # ============================================================

    def _initialize_agents(self):

        # --------------------------------------------------------
        # 1. FATYY LIVER
        # --------------------------------------------------------

        if self.fatty_agent is None:

            try:

                if FattyLiverAgent is None:
                    raise ImportError(
                        "FattyLiverAgent could not be imported."
                    )

                self._check_model_path(
                    FATTY_MODEL_PATH,
                    "Fatty Liver"
                )

                fatty_model_package = joblib.load(
                    FATTY_MODEL_PATH
                )

                self.fatty_agent = FattyLiverAgent(
                    fatty_model_package
                )

            except Exception as exc:

                self.fatty_agent = None

                self.initialization_errors[
                    "fatty_liver"
                ] = self._format_exception(exc)

        # --------------------------------------------------------
        # 2. FIBROSIS
        # --------------------------------------------------------

        if self.fibrosis_agent is None:

            try:

                if FibrosisAgent is None:
                    raise ImportError(
                        "FibrosisAgent could not be imported."
                    )

                self._check_model_path(
                    FIBROSIS_MODEL_PATH,
                    "Fibrosis"
                )

                fibrosis_model = joblib.load(
                    FIBROSIS_MODEL_PATH
                )

                self.fibrosis_agent = FibrosisAgent(
                    fibrosis_model
                )

            except Exception as exc:

                self.fibrosis_agent = None

                self.initialization_errors[
                    "fibrosis"
                ] = self._format_exception(exc)

        # --------------------------------------------------------
        # 3. CIRRHOSIS
        # --------------------------------------------------------

        if self.cirrhosis_agent is None:

            try:

                if CirrhosisAgent is None:
                    raise ImportError(
                        "CirrhosisAgent could not be imported."
                    )

                self._check_model_path(
                    CIRRHOSIS_MODEL_PATH,
                    "Cirrhosis"
                )

                self.cirrhosis_agent = CirrhosisAgent(
                    CIRRHOSIS_MODEL_PATH
                )

            except Exception as exc:

                self.cirrhosis_agent = None

                self.initialization_errors[
                    "cirrhosis"
                ] = self._format_exception(exc)

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

                self._check_model_path(
                    TUMOR_MODEL_PATH,
                    "Tumor Classification"
                )

                self.tumor_agent = (
                    TumorClassificationAgent(
                        TUMOR_MODEL_PATH
                    )
                )

            except Exception as exc:

                self.tumor_agent = None

                self.initialization_errors[
                    "tumor_classification"
                ] = self._format_exception(exc)

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

                self._check_model_path(
                    SEGMENTATION_MODEL_PATH,
                    "Liver Segmentation"
                )

                self.segmentation_agent = (
                    LiverSegmentationAgent(
                        SEGMENTATION_MODEL_PATH,
                        device=self.device,
                        target_size=(
                            128,
                            128,
                            64
                        ),
                        threshold=0.5,
                    )
                )

            except Exception as exc:

                self.segmentation_agent = None

                self.initialization_errors[
                    "liver_segmentation"
                ] = self._format_exception(exc)

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

                self._check_model_path(
                    CLINICAL_MODEL_PATH,
                    "Clinical Reasoning"
                )

                self.clinical_reasoning_agent = (
                    ClinicalReasoningAgent(
                        CLINICAL_MODEL_PATH
                    )
                )

            except Exception as exc:

                self.clinical_reasoning_agent = None

                self.initialization_errors[
                    "clinical_reasoning"
                ] = self._format_exception(exc)

    # ============================================================
    # MODEL PATH CHECK
    # ============================================================

    @staticmethod
    def _check_model_path(
        path: str,
        model_name: str
    ):

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"{model_name} model not found: {path}"
            )

    # ============================================================
    # PUBLIC API
    # ============================================================

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

        Example:

        patient_data = {
            "fatty_liver": {...},
            "fibrosis": {...},
            "cirrhosis": {...},
            "tumor": image,
            "segmentation": volume,
            "clinical_reasoning": {...}
        }

        result = orchestrator.run(
            patient_id="P001",
            patient_data=patient_data
        )
        """

        start_time = time.perf_counter()

        if patient_data is None:
            patient_data = {}
        else:
            # Avoid modifying the caller's dictionary
            patient_data = dict(patient_data)

        # --------------------------------------------------------
        # Legacy arguments
        # --------------------------------------------------------

        if clinical_data is not None:

            if "fatty_liver" not in patient_data:
                patient_data["fatty_liver"] = clinical_data

            if "clinical_reasoning" not in patient_data:
                patient_data["clinical_reasoning"] = clinical_data

        if fibrosis_input is not None:
            patient_data["fibrosis"] = fibrosis_input

        if cirrhosis_input is not None:
            patient_data["cirrhosis"] = cirrhosis_input

        if image is not None:
            patient_data["tumor"] = image

        if volume is not None:
            patient_data["segmentation"] = volume

        if clinical_reasoning_input is not None:
            patient_data[
                "clinical_reasoning"
            ] = clinical_reasoning_input

        # --------------------------------------------------------
        # Specialized agents
        # --------------------------------------------------------

        specialized_results = (
            self.run_specialized_agents(
                patient_data
            )
        )

        # --------------------------------------------------------
        # Clinical reasoning
        # --------------------------------------------------------

        clinical_result = (
            self.run_clinical_reasoning(
                patient_data,
                specialized_results
            )
        )

        # --------------------------------------------------------
        # All results
        # --------------------------------------------------------

        all_results = list(
            specialized_results
        )

        if clinical_result is not None:
            all_results.append(
                clinical_result
            )

        # --------------------------------------------------------
        # Adaptive fusion
        # --------------------------------------------------------

        fusion_result = (
            self._run_adaptive_fusion(
                all_results
            )
        )

        # --------------------------------------------------------
        # Conflict detection
        # --------------------------------------------------------

        conflicts = (
            self._run_conflict_detection(
                all_results
            )
        )

        # --------------------------------------------------------
        # Decision
        # --------------------------------------------------------

        decision = (
            self._run_decision_engine(
                all_results,
                conflicts,
                clinical_result
            )
        )

        # --------------------------------------------------------
        # Status
        # --------------------------------------------------------

        success_count = sum(
            1
            for result in all_results
            if result.get("status") == "success"
        )

        total_agents = len(
            self.agent_registry
        )

        if success_count == 0:

            overall_status = "failed"

        elif success_count < total_agents:

            overall_status = "partial"

        else:

            overall_status = "success"

        # --------------------------------------------------------
        # Latency
        # --------------------------------------------------------

        total_latency = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        # --------------------------------------------------------
        # Final result
        # --------------------------------------------------------

        return {

            "patient_id": patient_id,

            "timestamp":
                datetime.utcnow().isoformat(),

            "status":
                overall_status,

            "agents": {

                "fatty_liver":
                    self._find_result(
                        all_results,
                        "fatty_liver"
                    ),

                "fibrosis":
                    self._find_result(
                        all_results,
                        "fibrosis"
                    ),

                "cirrhosis":
                    self._find_result(
                        all_results,
                        "cirrhosis"
                    ),

                "tumor_classification":
                    self._find_result(
                        all_results,
                        "tumor_classification"
                    ),

                "liver_segmentation":
                    self._find_result(
                        all_results,
                        "liver_segmentation"
                    ),

                "clinical_reasoning":
                    self._find_result(
                        all_results,
                        "clinical_reasoning"
                    ),
            },

            "fusion":
                fusion_result,

            "conflicts":
                conflicts,

            "decision":
                decision,

            "summary": {

                "successful_agents":
                    success_count,

                "total_agents":
                    total_agents,

                "coverage":
                    (
                        success_count
                        / total_agents
                        if total_agents > 0
                        else 0.0
                    ),

                "latency_ms":
                    round(
                        total_latency,
                        2
                    ),
            },

            "initialization_errors":
                self.initialization_errors,
        }

    # ============================================================
    # ANALYZE ALIAS
    # ============================================================

    def analyze(
        self,
        patient_id,
        patient_data=None,
        **kwargs
    ):

        return self.run(
            patient_id=patient_id,
            patient_data=patient_data,
            **kwargs
        )

    # ============================================================
    # BACKWARD COMPATIBILITY
    # ============================================================

    def predict(
        self,
        patient_id,
        clinical_data=None,
        fibrosis_input=None,
        image=None,
        volume=None,
        patient_data=None,
        **kwargs
    ):

        return self.run(

            patient_id=patient_id,

            patient_data=patient_data,

            clinical_data=clinical_data,

            fibrosis_input=fibrosis_input,

            image=image,

            volume=volume,

            **kwargs
        )

    # ============================================================
    # SPECIALIZED AGENTS
    # ============================================================

    def run_specialized_agents(
        self,
        patient_data
    ):

        results = []

        agent_inputs = {

            "fatty_liver":
                patient_data.get(
                    "fatty_liver"
                ),

            "fibrosis":
                patient_data.get(
                    "fibrosis"
                ),

            "cirrhosis":
                patient_data.get(
                    "cirrhosis"
                ),

            "tumor_classification":
                patient_data.get(
                    "tumor"
                ),

            "liver_segmentation":
                patient_data.get(
                    "segmentation"
                ),
        }

        task_types = {

            "fatty_liver":
                "fatty_liver",

            "fibrosis":
                "fibrosis",

            "cirrhosis":
                "cirrhosis",

            "tumor_classification":
                "tumor_classification",

            "liver_segmentation":
                "liver_segmentation",
        }

        modalities = {

            "fatty_liver":
                "tabular",

            "fibrosis":
                "tabular",

            "cirrhosis":
                "tabular",

            "tumor_classification":
                "2d_image",

            "liver_segmentation":
                "3d_volume",
        }

        agents = [

            (
                "fatty_liver",
                self.fatty_agent
            ),

            (
                "fibrosis",
                self.fibrosis_agent
            ),

            (
                "cirrhosis",
                self.cirrhosis_agent
            ),

            (
                "tumor_classification",
                self.tumor_agent
            ),

            (
                "liver_segmentation",
                self.segmentation_agent
            ),
        ]

        for agent_id, agent in agents:

            input_data = agent_inputs.get(
                agent_id
            )

            # ----------------------------------------------------
            # Agent unavailable
            # ----------------------------------------------------

            if agent is None:

                reason = (
                    self.initialization_errors.get(
                        agent_id,
                        "Agent not initialized"
                    )
                )

                results.append(
                    self._not_run_result(
                        agent_id,
                        task_types[agent_id],
                        reason
                    )
                )

                continue

            # ----------------------------------------------------
            # Input missing
            # ----------------------------------------------------

            if input_data is None:

                results.append(
                    self._not_run_result(
                        agent_id,
                        task_types[agent_id],
                        "Required input not provided"
                    )
                )

                continue

            # ----------------------------------------------------
            # Execute
            # ----------------------------------------------------

            result = self._execute_agent(

                agent=agent,

                agent_id=agent_id,

                task_type=
                    task_types[agent_id],

                input_data=input_data,

                modality=
                    modalities[agent_id],
            )

            results.append(result)

        return results

    # ============================================================
    # CLINICAL REASONING
    # ============================================================

    def run_clinical_reasoning(
        self,
        patient_data,
        specialized_results
    ):

        agent = (
            self.clinical_reasoning_agent
        )

        if agent is None:

            reason = (
                self.initialization_errors.get(
                    "clinical_reasoning",
                    "Clinical reasoning agent not initialized"
                )
            )

            return self._not_run_result(
                "clinical_reasoning",
                "clinical_reasoning",
                reason
            )

        # --------------------------------------------------------
        # Correct input for the trained model
        # --------------------------------------------------------

        clinical_input = (
            patient_data.get(
                "clinical_reasoning"
            )
        )

        if clinical_input is None:

            clinical_input = (
                patient_data.get(
                    "fatty_liver"
                )
            )

        if clinical_input is None:

            return self._not_run_result(
                "clinical_reasoning",
                "clinical_reasoning",
                "Clinical reasoning input not provided"
            )

        # --------------------------------------------------------
        # Execute clinical model
        # --------------------------------------------------------

        result = self._execute_agent(

            agent=agent,

            agent_id=
                "clinical_reasoning",

            task_type=
                "clinical_reasoning",

            input_data=
                clinical_input,

            modality=
                "tabular",
        )

        # --------------------------------------------------------
        # Attach specialized evidence
        # --------------------------------------------------------

        evidence = {}

        for r in specialized_results:

            if r.get("status") == "success":

                evidence[
                    r.get("agent_id")
                ] = {

                    "prediction":
                        r.get("prediction"),

                    "probability":
                        r.get("probability"),

                    "class_probabilities":
                        r.get(
                            "class_probabilities"
                        ),

                    "confidence":
                        r.get("confidence"),

                    "uncertainty":
                        r.get("uncertainty"),

                    "trust":
                        r.get("trust"),
                }

        result["evidence"] = evidence

        return result

    # ============================================================
    # AGENT EXECUTION
    # ============================================================

    def _execute_agent(
        self,
        agent,
        agent_id,
        task_type,
        input_data,
        modality
    ):

        start = time.perf_counter()

        try:

            output = None

            # ----------------------------------------------------
            # predict()
            # ----------------------------------------------------

            if hasattr(agent, "predict"):

                try:

                    output = agent.predict(
                        input_data
                    )

                except TypeError:

                    try:

                        output = agent.predict(
                            data=input_data
                        )

                    except TypeError:

                        output = agent.predict(
                            input_data
                        )

            # ----------------------------------------------------
            # analyze()
            # ----------------------------------------------------

            elif hasattr(agent, "analyze"):

                output = agent.analyze(
                    input_data
                )

            # ----------------------------------------------------
            # run()
            # ----------------------------------------------------

            elif hasattr(agent, "run"):

                output = agent.run(
                    input_data
                )

            else:

                raise AttributeError(
                    f"Agent {agent_id} has no "
                    "predict(), analyze() or run() method."
                )

            latency = (
                time.perf_counter()
                - start
            ) * 1000.0

            return self._normalize_result(

                output=output,

                agent_id=agent_id,

                task_type=task_type,

                modality=modality,

                latency=latency,
            )

        except Exception as exc:

            latency = (
                time.perf_counter()
                - start
            ) * 1000.0

            return {

                "agent_id":
                    agent_id,

                "task_type":
                    task_type,

                "modality":
                    modality,

                "prediction":
                    None,

                "probability":
                    None,

                "class_probabilities":
                    None,

                "confidence":
                    0.0,

                "uncertainty":
                    1.0,

                "quality":
                    0.0,

                "trust":
                    0.0,

                "latency_ms":
                    round(
                        latency,
                        2
                    ),

                "missing_data_ratio":
                    1.0,

                "status":
                    "failed",

                "details":
                    {},

                "explanation":
                    "",

                "error":
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
            }

    # ============================================================
    # RESULT NORMALIZATION
    # ============================================================

    def _normalize_result(
        self,
        output,
        agent_id,
        task_type,
        modality,
        latency
    ):

        if output is None:
            output = {}

        if not isinstance(output, dict):

            output = {
                "prediction": output
            }

        prediction = output.get(
            "prediction",
            output.get(
                "predicted_class"
            )
        )

        probability = output.get(
            "probability"
        )

        if probability is None:

            probability = output.get(
                "confidence"
            )

        class_probabilities = (
            output.get(
                "class_probabilities"
            )
        )

        if class_probabilities is None:

            class_probabilities = (
                output.get(
                    "probabilities"
                )
            )

        # --------------------------------------------------------
        # Confidence
        # --------------------------------------------------------

        raw_confidence = output.get(
            "confidence"
        )

        if raw_confidence is None:

            raw_confidence = (
                self._extract_probability_confidence(
                    class_probabilities
                )
            )

        if raw_confidence is None:

            raw_confidence = (
                self._extract_probability_confidence(
                    probability
                )
            )

        confidence = self._clip(
            self._safe_float(
                raw_confidence,
                0.0
            )
        )

        # --------------------------------------------------------
        # Uncertainty
        # --------------------------------------------------------

        raw_uncertainty = output.get(
            "uncertainty"
        )

        if raw_uncertainty is None:

            raw_uncertainty = (
                1.0 - confidence
            )

        uncertainty = self._clip(
            self._safe_float(
                raw_uncertainty,
                1.0 - confidence
            )
        )

        # --------------------------------------------------------
        # Quality
        # --------------------------------------------------------

        quality = self._clip(
            self._safe_float(
                output.get(
                    "quality",
                    1.0
                ),
                1.0
            )
        )

        # --------------------------------------------------------
        # Missing data
        # --------------------------------------------------------

        missing_ratio = self._clip(
            self._safe_float(
                output.get(
                    "missing_data_ratio",
                    0.0
                ),
                0.0
            )
        )

        # --------------------------------------------------------
        # Status
        # --------------------------------------------------------

        status = output.get(
            "status",
            "success"
        )

        if status is None:
            status = "success"

        # --------------------------------------------------------
        # Trust
        # --------------------------------------------------------

        trust = self._compute_trust(

            agent_id=
                agent_id,

            confidence=
                confidence,

            quality=
                quality,

            uncertainty=
                uncertainty,

            missing_data_ratio=
                missing_ratio,
        )

        # --------------------------------------------------------
        # Normalized result
        # --------------------------------------------------------

        result = {

            "agent_id":
                agent_id,

            "task_type":
                task_type,

            "modality":
                modality,

            "prediction":
                prediction,

            "probability":
                probability,

            "class_probabilities":
                class_probabilities,

            "confidence":
                confidence,

            "uncertainty":
                uncertainty,

            "quality":
                quality,

            "trust":
                trust,

            "latency_ms":
                round(
                    latency,
                    2
                ),

            "missing_data_ratio":
                missing_ratio,

            "status":
                status,

            "details":
                output.get(
                    "details",
                    {}
                ),

            "explanation":
                output.get(
                    "explanation",
                    ""
                ),

            "error":
                output.get(
                    "error"
                ),
        }

        # --------------------------------------------------------
        # Preserve model-specific fields
        # --------------------------------------------------------

        for key, value in output.items():

            if key not in result:

                result[key] = value

        return result

    # ============================================================
    # CONFIDENCE FROM PROBABILITIES
    # ============================================================

    def _extract_probability_confidence(
        self,
        probability
    ):

        if probability is None:
            return None

        # Dictionary
        if isinstance(
            probability,
            dict
        ):

            values = []

            for value in probability.values():

                try:
                    values.append(
                        float(value)
                    )
                except Exception:
                    continue

            if values:
                return max(values)

            return None

        # List / tuple
        if isinstance(
            probability,
            (list, tuple)
        ):

            values = []

            for value in probability:

                try:
                    values.append(
                        float(value)
                    )
                except Exception:
                    continue

            if values:
                return max(values)

            return None

        # Scalar
        try:

            return float(
                probability
            )

        except Exception:

            return None

    # ============================================================
    # TRUST
    # ============================================================

    def _compute_trust(
        self,
        agent_id,
        confidence,
        quality,
        uncertainty,
        missing_data_ratio
    ):

        if self.trust_manager is None:

            trust = (

                0.45 * confidence

                + 0.30 * quality

                + 0.15
                * (1.0 - uncertainty)

                + 0.10
                * (1.0 - missing_data_ratio)
            )

            return self._clip(
                trust
            )

        try:

            trust = (
                self.trust_manager.compute_trust(

                    agent_id=
                        agent_id,

                    confidence=
                        confidence,

                    quality=
                        quality,

                    uncertainty=
                        uncertainty,

                    missing_data_ratio=
                        missing_data_ratio,

                    agreement=
                        0.5,

                    stability=
                        0.5,

                    utility=
                        0.5,

                    modality_available=
                        True,
                )
            )

            return self._clip(
                trust
            )

        except Exception:

            trust = (

                0.45 * confidence

                + 0.30 * quality

                + 0.15
                * (1.0 - uncertainty)

                + 0.10
                * (1.0 - missing_data_ratio)
            )

            return self._clip(
                trust
            )

    # ============================================================
    # ADAPTIVE FUSION
    # ============================================================

    def _run_adaptive_fusion(
        self,
        results
    ):

        if not results:

            return {
                "status":
                    "no_valid_results",

                "evidence":
                    [],

                "weights":
                    {},

                "task_groups":
                    {},

                "same_task_fusion":
                    {},
            }

        if self.adaptive_fusion is None:

            return {

                "status":
                    "unavailable",

                "results":
                    results,
            }

        # Only successful agents
        valid_results = [

            result

            for result in results

            if result.get(
                "status"
            ) == "success"
        ]

        if not valid_results:

            return {

                "status":
                    "no_valid_results",

                "evidence":
                    [],

                "weights":
                    {},

                "task_groups":
                    {},

                "same_task_fusion":
                    {},
            }

        try:

            return self.adaptive_fusion.fuse(
                valid_results
            )

        except Exception as exc:

            return {

                "status":
                    "failed",

                "error":
                    str(exc),

                "results":
                    valid_results,
            }

    # ============================================================
    # CONFLICT DETECTION
    # ============================================================

    def _run_conflict_detection(
        self,
        results
    ):

        if self.conflict_detector is None:
            return []

        if AgentResult is None:
            return []

        objects = []

        for result in results:

            if result.get(
                "status"
            ) != "success":

                continue

            try:

                objects.append(
                    AgentResult.from_dict(
                        result
                    )
                )

            except Exception:
                continue

        if not objects:
            return []

        try:

            return self.conflict_detector.detect(
                objects
            )

        except Exception as exc:

            return [

                {
                    "type":
                        "conflict_detection_error",

                    "error":
                        str(exc),
                }
            ]

    # ============================================================
    # DECISION ENGINE
    # ============================================================

    def _run_decision_engine(
        self,
        results,
        conflicts,
        clinical_result
    ):

        if self.decision_engine is None:

            return {
                "status":
                    "unavailable"
            }

        clinical_dict = (

            clinical_result

            if isinstance(
                clinical_result,
                dict
            )

            else {}
        )

        try:

            return self.decision_engine.decide(

                results,

                conflicts,

                clinical_dict
            )

        except Exception as exc:

            return {

                "status":
                    "failed",

                "error":
                    str(exc),
            }

    # ============================================================
    # NOT RUN RESULT
    # ============================================================

    def _not_run_result(
        self,
        agent_id,
        task_type,
        reason
    ):

        modality = {

            "fatty_liver":
                "tabular",

            "fibrosis":
                "tabular",

            "cirrhosis":
                "tabular",

            "tumor_classification":
                "2d_image",

            "liver_segmentation":
                "3d_volume",

            "clinical_reasoning":
                "tabular",
        }.get(
            agent_id
        )

        return {

            "agent_id":
                agent_id,

            "task_type":
                task_type,

            "modality":
                modality,

            "prediction":
                None,

            "probability":
                None,

            "class_probabilities":
                None,

            "confidence":
                0.0,

            "uncertainty":
                1.0,

            "quality":
                0.0,

            "trust":
                0.0,

            "latency_ms":
                0.0,

            "missing_data_ratio":
                1.0,

            "status":
                "not_run",

            "details":
                {},

            "explanation":
                "",

            "error":
                reason,
        }

    # ============================================================
    # FIND RESULT
    # ============================================================

    @staticmethod
    def _find_result(
        results,
        agent_id
    ):

        for result in results:

            if (
                result.get(
                    "agent_id"
                )
                == agent_id
            ):

                return result

        return {

            "agent_id":
                agent_id,

            "status":
                "not_run",
        }

    # ============================================================
    # SAFE FLOAT
    # ============================================================

    @staticmethod
    def _safe_float(
        value,
        default=0.0
    ):

        try:

            if value is None:
                return default

            value = float(value)

            if value != value:
                return default

            return value

        except Exception:

            return default

    # ============================================================
    # CLIP
    # ============================================================

    @staticmethod
    def _clip(
        value
    ):

        try:

            value = float(value)

        except Exception:

            value = 0.0

        return max(
            0.0,
            min(
                1.0,
                value
            )
        )

    # ============================================================
    # EXCEPTION FORMAT
    # ============================================================

    @staticmethod
    def _format_exception(
        exc
    ):

        return (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def health_check(self):

        agents = {

            "fatty_liver":
                self.fatty_agent,

            "fibrosis":
                self.fibrosis_agent,

            "cirrhosis":
                self.cirrhosis_agent,

            "tumor_classification":
                self.tumor_agent,

            "liver_segmentation":
                self.segmentation_agent,

            "clinical_reasoning":
                self.clinical_reasoning_agent,
        }

        initialized = sum(
            agent is not None
            for agent in agents.values()
        )

        return {

            "status":
                (
                    "healthy"
                    if initialized == 6
                    else (
                        "partial"
                        if initialized > 0
                        else "failed"
                    )
                ),

            "device":
                self.device,

            "initialized_agents":
                initialized,

            "total_agents":
                6,

            "agents": {
                name:
                    agent is not None
                for name, agent
                in agents.items()
            },

            "initialization_errors":
                self.initialization_errors,

            "coordinators": {

                "trust_manager":
                    self.trust_manager is not None,

                "adaptive_fusion":
                    self.adaptive_fusion is not None,

                "conflict_detector":
                    self.conflict_detector is not None,

                "decision_engine":
                    self.decision_engine is not None,
            },
        }
