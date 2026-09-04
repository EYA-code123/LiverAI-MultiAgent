# =============================================================================
# LiverAI-MultiAgent
# COMPLETE MULTI-AGENT ORCHESTRATOR
# =============================================================================
#
# Architecture
#
#                   PATIENT DATA
#                        |
#                        v
#               LIVER AI ORCHESTRATOR
#                        |
#       +----------------+----------------+
#       |        |        |       |       |
#       v        v        v       v       v
#     Fatty   Fibrosis Cirrhosis Tumor  Segmentation
#       |        |        |       |       |
#       +--------+--------+-------+-------+
#                        |
#                        v
#                  TRUST MANAGER
#                        |
#                        v
#                 ADAPTIVE FUSION
#                        |
#                        v
#                CONFLICT DETECTOR
#                        |
#                        v
#               CLINICAL REASONING
#                        |
#                        v
#                 DECISION ENGINE
#                        |
#                        v
#                  FINAL REPORT
#
# =============================================================================

import os
import time
import traceback
from datetime import datetime

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


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class LiverAIOrchestrator:
    """
    Main orchestration layer for the LiverAI Multi-Agent system.

    Six agents are coordinated:

        1. Fatty Liver Agent
        2. Fibrosis Agent
        3. Cirrhosis Agent
        4. Tumor Classification Agent
        5. Liver Segmentation Agent
        6. Clinical Reasoning Agent

    Input structure
    ---------------

    patient_data = {
        "fatty_liver": {
            ...
        },

        "fibrosis": {
            ...
        },

        "cirrhosis": {
            ...
        },

        "tumor": image,

        "segmentation": volume,

        "clinical_reasoning": {
            ...
        }
    }

    The orchestrator supports partial multimodal input.

    Example:

        result = orchestrator.run(
            patient_id="PATIENT_001",
            patient_data=patient_data
        )

    """

    # =========================================================================
    # DEFAULT MODEL PATHS
    # =========================================================================

    DEFAULT_MODEL_PATHS = {

        "fatty_liver": (
            "/content/drive/MyDrive/"
            "Fatty_Liver_Dataset/models/"
            "FattyLiver_LightGBM.pkl"
        ),

        "fibrosis": (
            "/content/drive/MyDrive/"
            "Fibrosis Agent/XGBoost_model/"
            "xgboost_nafld.pkl"
        ),

        "cirrhosis": (
            "/content/drive/MyDrive/"
            ".Cirrhosis Agent/XGBoost_model/"
            "XGBoost_Cirrhosis_fixed.joblib"
        ),

        "tumor_classification": (
            "/content/drive/MyDrive/"
            "models/tumor/"
            "efficientnet_b0_best.pth"
        ),

        "liver_segmentation": (
            "/content/drive/MyDrive/"
            "Liver Segmentation Agent/models/"
            "SegResNet3D_Liver_best.pth"
        ),

        "clinical_reasoning": (
            "/content/drive/MyDrive/"
            "Clinical Reasoning Agent/"
            "tabtransformer_bupa"
        ),
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
    # CONSTRUCTOR
    # =========================================================================

    def __init__(
        self,
        model_paths=None,
        device=None,
        verbose=True,
    ):
        """
        Initialize all LiverAI agents.

        Parameters
        ----------
        model_paths : dict, optional
            Custom model paths.

        device : str, optional
            Device forwarded when supported.

        verbose : bool
            Print initialization information.
        """

        self.name = "LiverAI Multi-Agent Orchestrator"

        self.verbose = verbose

        # ---------------------------------------------------------------------
        # Merge default and custom paths
        # ---------------------------------------------------------------------

        self.model_paths = dict(self.DEFAULT_MODEL_PATHS)

        if model_paths is not None:
            self.model_paths.update(model_paths)

        self.device = device

        # ---------------------------------------------------------------------
        # Agent objects
        # ---------------------------------------------------------------------

        self.fatty_liver_agent = None
        self.fibrosis_agent = None
        self.cirrhosis_agent = None
        self.tumor_agent = None
        self.segmentation_agent = None
        self.clinical_agent = None

        # ---------------------------------------------------------------------
        # Coordination modules
        # ---------------------------------------------------------------------

        self.trust_manager = None
        self.adaptive_fusion = None
        self.conflict_detector = None
        self.decision_engine = None

        # ---------------------------------------------------------------------
        # Load agents
        # ---------------------------------------------------------------------

        self._load_fatty_liver_agent()
        self._load_fibrosis_agent()
        self._load_cirrhosis_agent()
        self._load_tumor_agent()
        self._load_segmentation_agent()
        self._load_clinical_reasoning_agent()

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
        # Initialize coordination components
        # ---------------------------------------------------------------------

        try:
            if TrustManager is not None:
                self.trust_manager = TrustManager()

            if AdaptiveFusion is not None:
                self.adaptive_fusion = AdaptiveFusion()

            if ConflictDetector is not None:
                self.conflict_detector = ConflictDetector()

            if DecisionEngine is not None:
                self.decision_engine = DecisionEngine()

        except Exception as exc:
            self._log(
                "WARNING",
                f"Coordination initialization issue: "
                f"{type(exc).__name__}: {exc}",
            )

        self._print_initialization_summary()

    # =========================================================================
    # LOGGING
    # =========================================================================

    def _log(self, level, message):
        """Print orchestrator logs."""

        if self.verbose:
            print(f"[LiverAI][{level}] {message}")

    # =========================================================================
    # AGENT LOADERS
    # =========================================================================

    def _load_fatty_liver_agent(self):
        """Load Fatty Liver Agent."""

        try:
            import joblib

            from agents.fatty_liver_agent import FattyLiverAgent

            path = self.model_paths["fatty_liver"]

            if not os.path.exists(path):
                raise FileNotFoundError(path)

            model = joblib.load(path)

            self.fatty_liver_agent = FattyLiverAgent(model)

            self._log(
                "INFO",
                "✓ Fatty Liver Agent loaded",
            )

        except Exception as exc:

            self.fatty_liver_agent = None

            self._log(
                "ERROR",
                f"✗ Fatty Liver Agent failed: "
                f"{type(exc).__name__}: {exc}",
            )

    # -------------------------------------------------------------------------

    def _load_fibrosis_agent(self):
        """Load Fibrosis Agent."""

        try:
            import joblib

            from agents.fibrosis_agent import FibrosisAgent

            path = self.model_paths["fibrosis"]

            if not os.path.exists(path):
                raise FileNotFoundError(path)

            model = joblib.load(path)

            self.fibrosis_agent = FibrosisAgent(model)

            self._log(
                "INFO",
                "✓ Fibrosis Agent loaded",
            )

        except Exception as exc:

            self.fibrosis_agent = None

            self._log(
                "ERROR",
                f"✗ Fibrosis Agent failed: "
                f"{type(exc).__name__}: {exc}",
            )

    # -------------------------------------------------------------------------

    def _load_cirrhosis_agent(self):
        """Load Cirrhosis Agent."""

        try:

            from agents.cirrhosis_agent import CirrhosisAgent

            path = self.model_paths["cirrhosis"]

            if not os.path.exists(path):
                raise FileNotFoundError(path)

            self.cirrhosis_agent = CirrhosisAgent(path)

            self._log(
                "INFO",
                "✓ Cirrhosis Agent loaded",
            )

        except Exception as exc:

            self.cirrhosis_agent = None

            self._log(
                "ERROR",
                f"✗ Cirrhosis Agent failed: "
                f"{type(exc).__name__}: {exc}",
            )

    # -------------------------------------------------------------------------

    def _load_tumor_agent(self):
        """Load Tumor Classification Agent."""

        try:

            from agents.tumor_classification_agent import (
                TumorClassificationAgent
            )

            path = self.model_paths["tumor_classification"]

            if not os.path.exists(path):
                raise FileNotFoundError(path)

            # Device is passed only if explicitly specified.
            if self.device is not None:

                try:

                    self.tumor_agent = TumorClassificationAgent(
                        path,
                        device=self.device,
                    )

                except TypeError:

                    self.tumor_agent = TumorClassificationAgent(
                        path
                    )

            else:

                self.tumor_agent = TumorClassificationAgent(path)

            self._log(
                "INFO",
                "✓ Tumor Classification Agent loaded",
            )

        except Exception as exc:

            self.tumor_agent = None

            self._log(
                "ERROR",
                f"✗ Tumor Classification Agent failed: "
                f"{type(exc).__name__}: {exc}",
            )

    # -------------------------------------------------------------------------

    def _load_segmentation_agent(self):
        """Load Liver Segmentation Agent."""

        try:

            from agents.liver_segmentation_agent import (
                LiverSegmentationAgent
            )

            path = self.model_paths["liver_segmentation"]

            if not os.path.exists(path):
                raise FileNotFoundError(path)

            kwargs = {
                "model_path": path,
            }

            if self.device is not None:
                kwargs["device"] = self.device

            self.segmentation_agent = LiverSegmentationAgent(
                **kwargs
            )

            self._log(
                "INFO",
                "✓ Liver Segmentation Agent loaded",
            )

        except Exception as exc:

            self.segmentation_agent = None

            self._log(
                "ERROR",
                f"✗ Liver Segmentation Agent failed: "
                f"{type(exc).__name__}: {exc}",
            )

    # -------------------------------------------------------------------------

    def _load_clinical_reasoning_agent(self):
        """Load Clinical Reasoning Agent."""

        try:

            from agents.clinical_reasoning_agent import (
                ClinicalReasoningAgent
            )

            path = self.model_paths["clinical_reasoning"]

            if not os.path.exists(path):
                raise FileNotFoundError(path)

            self.clinical_agent = ClinicalReasoningAgent(path)

            self._log(
                "INFO",
                "✓ Clinical Reasoning Agent loaded",
            )

        except Exception as exc:

            self.clinical_agent = None

            self._log(
                "ERROR",
                f"✗ Clinical Reasoning Agent failed: "
                f"{type(exc).__name__}: {exc}",
            )

    # =========================================================================
    # INITIALIZATION SUMMARY
    # =========================================================================

    def _print_initialization_summary(self):
        """Print agent and coordinator status."""

        if not self.verbose:
            return

        print()
        print("=" * 78)
        print("LIVERAI MULTI-AGENT ORCHESTRATOR")
        print("=" * 78)

        for agent_id, agent in self.agents.items():

            status = "READY" if agent is not None else "NOT AVAILABLE"

            print(
                f"{agent_id:<25} : {status}"
            )

        print("-" * 78)

        coordination = {
            "TrustManager": self.trust_manager,
            "AdaptiveFusion": self.adaptive_fusion,
            "ConflictDetector": self.conflict_detector,
            "DecisionEngine": self.decision_engine,
        }

        for name, component in coordination.items():

            status = "READY" if component is not None else "NOT AVAILABLE"

            print(
                f"{name:<25} : {status}"
            )

        print("=" * 78)
        print()

    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================

    @staticmethod
    def _is_mapping(value):
        """Return True for dictionary-like values."""

        return isinstance(value, dict)

    # -------------------------------------------------------------------------

    @staticmethod
    def _has_value(value):
        """Check whether an input value is actually available."""

        if value is None:
            return False

        if isinstance(value, dict):
            return len(value) > 0

        return True

    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_nested(patient_data, key):
        """Safely extract a nested modality."""

        if not isinstance(patient_data, dict):
            return None

        return patient_data.get(key)

    # -------------------------------------------------------------------------

    @staticmethod
    def _safe_float(value, default=0.0):
        """Convert a value safely to float."""

        try:
            return float(value)
        except Exception:
            return float(default)

    # -------------------------------------------------------------------------

    @staticmethod
    def _safe_probability(value):
        """Convert probability/confidence to a valid float."""

        if value is None:
            return None

        try:
            value = float(value)
        except Exception:
            return None

        if value != value:
            return None

        return max(0.0, min(1.0, value))

    # =========================================================================
    # INPUT EXTRACTION
    # =========================================================================

    def _extract_inputs(
        self,
        patient_data=None,
        clinical_data=None,
        fibrosis_input=None,
        cirrhosis_input=None,
        image=None,
        volume=None,
        clinical_reasoning_input=None,
    ):
        """
        Normalize all supported input formats.

        Nested patient_data has priority over individual arguments.
        """

        if patient_data is None:
            patient_data = {}

        if not isinstance(patient_data, dict):
            raise TypeError(
                "patient_data must be a dictionary."
            )

        # ---------------------------------------------------------------------
        # Fatty liver
        # ---------------------------------------------------------------------

        fatty_input = patient_data.get(
            "fatty_liver"
        )

        if fatty_input is None:
            fatty_input = clinical_data

        # ---------------------------------------------------------------------
        # Fibrosis
        # ---------------------------------------------------------------------

        fibrosis_data = patient_data.get(
            "fibrosis"
        )

        if fibrosis_data is None:
            fibrosis_data = fibrosis_input

        # ---------------------------------------------------------------------
        # Cirrhosis
        # ---------------------------------------------------------------------

        cirrhosis_data = patient_data.get(
            "cirrhosis"
        )

        if cirrhosis_data is None:
            cirrhosis_data = cirrhosis_input

        # ---------------------------------------------------------------------
        # Tumor
        # ---------------------------------------------------------------------

        tumor_image = patient_data.get(
            "tumor"
        )

        if tumor_image is None:
            tumor_image = patient_data.get(
                "tumor_classification"
            )

        if tumor_image is None:
            tumor_image = image

        # ---------------------------------------------------------------------
        # Segmentation
        # ---------------------------------------------------------------------

        segmentation_volume = patient_data.get(
            "segmentation"
        )

        if segmentation_volume is None:
            segmentation_volume = patient_data.get(
                "liver_segmentation"
            )

        if segmentation_volume is None:
            segmentation_volume = volume

        # ---------------------------------------------------------------------
        # Clinical reasoning
        # ---------------------------------------------------------------------

        reasoning_input = patient_data.get(
            "clinical_reasoning"
        )

        if reasoning_input is None:
            reasoning_input = clinical_reasoning_input

        if reasoning_input is None:
            reasoning_input = fatty_input

        return {
            "fatty_liver": fatty_input,
            "fibrosis": fibrosis_data,
            "cirrhosis": cirrhosis_data,
            "tumor": tumor_image,
            "segmentation": segmentation_volume,
            "clinical_reasoning": reasoning_input,
        }

    # =========================================================================
    # AGENT EXECUTION
    # =========================================================================

    def _call_agent(
        self,
        agent,
        input_data,
        agent_id,
    ):
        """
        Execute an agent using the API exposed by the agent.

        Supported method names:

            predict()
            run()
            analyze()

        The method is detected automatically.
        """

        if agent is None:
            raise RuntimeError(
                f"{agent_id} is not loaded."
            )

        if input_data is None:
            raise ValueError(
                f"No input data supplied for {agent_id}."
            )

        # ---------------------------------------------------------------------
        # Method priority
        # ---------------------------------------------------------------------

        methods = [
            "predict",
            "run",
            "analyze",
        ]

        for method_name in methods:

            method = getattr(
                agent,
                method_name,
                None
            )

            if callable(method):

                try:
                    return method(input_data)

                except TypeError:

                    # Some agents may use a named argument.
                    try:

                        return method(
                            input_data=input_data
                        )

                    except TypeError:

                        continue

        raise AttributeError(
            f"No compatible execution method found "
            f"for {agent_id}. "
            f"Expected predict(), run() or analyze()."
        )

    # =========================================================================
    # AGENT RESULT NORMALIZATION
    # =========================================================================

    def _normalize_agent_output(
        self,
        agent_id,
        task_type,
        raw_output,
        latency_ms,
        modality,
    ):
        """
        Convert arbitrary agent output into a standardized dictionary.

        This dictionary intentionally preserves additional metadata such as:

            modality
            class_probabilities

        because AdaptiveFusion uses these fields.
        """

        if isinstance(raw_output, AgentResult):

            result = raw_output.to_dict()

        elif isinstance(raw_output, dict):

            result = dict(raw_output)

        else:

            result = {
                "prediction": raw_output
            }

        # ---------------------------------------------------------------------
        # Standard fields
        # ---------------------------------------------------------------------

        result.setdefault(
            "agent_id",
            agent_id
        )

        result.setdefault(
            "task_type",
            task_type
        )

        result.setdefault(
            "status",
            "success"
        )

        result.setdefault(
            "modality",
            modality
        )

        result.setdefault(
            "latency_ms",
            latency_ms
        )

        # ---------------------------------------------------------------------
        # Normalize confidence
        # ---------------------------------------------------------------------

        confidence = (
            result.get("confidence")
        )

        if confidence is None:

            confidence = result.get(
                "probability"
            )

        confidence = self._safe_probability(
            confidence
        )

        if confidence is None:
            confidence = 0.0

        result["confidence"] = confidence

        # ---------------------------------------------------------------------
        # Probability
        # ---------------------------------------------------------------------

        if "probability" not in result:
            result["probability"] = confidence

        # ---------------------------------------------------------------------
        # Uncertainty
        # ---------------------------------------------------------------------

        if result.get("uncertainty") is None:

            result["uncertainty"] = (
                1.0 - confidence
            )

        result["uncertainty"] = self._safe_probability(
            result["uncertainty"]
        )

        # ---------------------------------------------------------------------
        # Quality
        # ---------------------------------------------------------------------

        if result.get("quality") is None:

            result["quality"] = 1.0

        result["quality"] = self._safe_probability(
            result["quality"]
        )

        # ---------------------------------------------------------------------
        # Missing data
        # ---------------------------------------------------------------------

        if result.get("missing_data_ratio") is None:

            result["missing_data_ratio"] = 0.0

        result["missing_data_ratio"] = self._safe_probability(
            result["missing_data_ratio"]
        )

        # ---------------------------------------------------------------------
        # Trust placeholder
        # ---------------------------------------------------------------------

        if result.get("trust") is None:
            result["trust"] = confidence

        result["trust"] = self._safe_probability(
            result["trust"]
        )

        # ---------------------------------------------------------------------
        # Class probabilities
        # ---------------------------------------------------------------------

        if "class_probabilities" not in result:

            if "probabilities" in result:

                result["class_probabilities"] = (
                    result["probabilities"]
                )

            elif "probs" in result:

                result["class_probabilities"] = (
                    result["probs"]
                )

        return result

    # =========================================================================
    # TRUST
    # =========================================================================

    def _compute_trust(
        self,
        agent_id,
        result,
    ):
        """
        Compute trust score using TrustManager.

        Falls back to confidence when TrustManager is unavailable.
        """

        confidence = self._safe_probability(
            result.get("confidence")
        )

        quality = self._safe_probability(
            result.get("quality")
        )

        uncertainty = self._safe_probability(
            result.get("uncertainty")
        )

        missing_ratio = self._safe_probability(
            result.get("missing_data_ratio")
        )

        if confidence is None:
            confidence = 0.0

        if quality is None:
            quality = 0.0

        if uncertainty is None:
            uncertainty = 1.0

        if missing_ratio is None:
            missing_ratio = 0.0

        # ---------------------------------------------------------------------
        # TrustManager available
        # ---------------------------------------------------------------------

        if self.trust_manager is not None:

            try:

                trust = self.trust_manager.compute_trust(
                    agent_id=agent_id,
                    confidence=confidence,
                    quality=quality,
                    uncertainty=uncertainty,
                    missing_data_ratio=missing_ratio,
                    agreement=0.5,
                    stability=0.5,
                    utility=0.5,
                    modality_available=True,
                )

                trust = self._safe_probability(
                    trust
                )

                if trust is not None:
                    return trust

            except Exception as exc:

                self._log(
                    "WARNING",
                    f"Trust computation failed for "
                    f"{agent_id}: {exc}",
                )

        # ---------------------------------------------------------------------
        # Fallback trust
        # ---------------------------------------------------------------------

        trust = (
            0.50 * confidence
            + 0.30 * quality
            + 0.20 * (1.0 - uncertainty)
        )

        return max(
            0.0,
            min(1.0, trust)
        )

    # =========================================================================
    # AGENT RESULT OBJECT
    # =========================================================================

    def _to_agent_result(
        self,
        result,
    ):
        """
        Convert standardized dictionary to AgentResult.

        If conversion fails, create a safe fallback.
        """

        try:

            return AgentResult.from_dict(
                result
            )

        except Exception as exc:

            self._log(
                "WARNING",
                "AgentResult conversion failed: "
                f"{type(exc).__name__}: {exc}",
            )

            # Minimal safe fallback
            return AgentResult(
                agent_id=result.get(
                    "agent_id",
                    "unknown"
                ),

                task_type=result.get(
                    "task_type",
                    "unknown"
                ),

                prediction=result.get(
                    "prediction"
                ),

                probability=result.get(
                    "probability"
                ),

                confidence=result.get(
                    "confidence",
                    0.0
                ),

                uncertainty=result.get(
                    "uncertainty",
                    1.0
                ),

                quality=result.get(
                    "quality",
                    0.0
                ),

                latency_ms=result.get(
                    "latency_ms",
                    0.0
                ),

                missing_data_ratio=result.get(
                    "missing_data_ratio",
                    0.0
                ),

                trust=result.get(
                    "trust",
                    0.0
                ),

                status=result.get(
                    "status",
                    "error"
                ),

                details=result.get(
                    "details"
                ),

                explanation=result.get(
                    "explanation"
                ),

                error=result.get(
                    "error"
                ),
            )

    # =========================================================================
    # GENERIC AGENT EXECUTION
    # =========================================================================

    def _execute_agent(
        self,
        agent_id,
        task_type,
        input_data,
        modality,
    ):
        """
        Execute one specialized agent.

        Returns
        -------
        dict
            Standardized result dictionary.
        """

        start_time = time.perf_counter()

        agent = self.agents.get(
            agent_id
        )

        # ---------------------------------------------------------------------
        # Agent unavailable
        # ---------------------------------------------------------------------

        if agent is None:

            return {
                "agent_id": agent_id,
                "task_type": task_type,
                "modality": modality,
                "prediction": None,
                "probability": None,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "latency_ms": 0.0,
                "missing_data_ratio": 1.0,
                "trust": 0.0,
                "status": "not_available",
                "details": {},
                "explanation": (
                    f"{agent_id} is not available."
                ),
                "error": (
                    f"{agent_id} failed to load."
                ),
            }

        # ---------------------------------------------------------------------
        # Missing input
        # ---------------------------------------------------------------------

        if not self._has_value(input_data):

            return {
                "agent_id": agent_id,
                "task_type": task_type,
                "modality": modality,
                "prediction": None,
                "probability": None,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "latency_ms": 0.0,
                "missing_data_ratio": 1.0,
                "trust": 0.0,
                "status": "not_run",
                "details": {},
                "explanation": (
                    f"No {modality} input supplied."
                ),
                "error": None,
            }

        # ---------------------------------------------------------------------
        # Execute
        # ---------------------------------------------------------------------

        try:

            raw_output = self._call_agent(
                agent=agent,
                input_data=input_data,
                agent_id=agent_id,
            )

            latency_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

            result = self._normalize_agent_output(
                agent_id=agent_id,
                task_type=task_type,
                raw_output=raw_output,
                latency_ms=latency_ms,
                modality=modality,
            )

            # Ensure successful status.
            result["status"] = "success"

            # Compute trust.
            result["trust"] = self._compute_trust(
                agent_id=agent_id,
                result=result,
            )

            return result

        except Exception as exc:

            latency_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

            error_text = (
                f"{type(exc).__name__}: {exc}"
            )

            self._log(
                "ERROR",
                f"{agent_id} execution failed: "
                f"{error_text}",
            )

            return {
                "agent_id": agent_id,
                "task_type": task_type,
                "modality": modality,
                "prediction": None,
                "probability": None,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "latency_ms": latency_ms,
                "missing_data_ratio": 1.0,
                "trust": 0.0,
                "status": "error",
                "details": {},
                "explanation": (
                    f"{agent_id} execution failed."
                ),
                "error": error_text,
                "traceback": traceback.format_exc(),
            }

    # =========================================================================
    # INDIVIDUAL AGENT METHODS
    # =========================================================================

    def run_fatty_liver(
        self,
        data,
    ):
        """Run Fatty Liver Agent."""

        return self._execute_agent(
            agent_id="fatty_liver",
            task_type="fatty_liver",
            input_data=data,
            modality="clinical_tabular",
        )

    # -------------------------------------------------------------------------

    def run_fibrosis(
        self,
        data,
    ):
        """Run Fibrosis Agent."""

        return self._execute_agent(
            agent_id="fibrosis",
            task_type="fibrosis",
            input_data=data,
            modality="clinical_tabular",
        )

    # -------------------------------------------------------------------------

    def run_cirrhosis(
        self,
        data,
    ):
        """Run Cirrhosis Agent."""

        return self._execute_agent(
            agent_id="cirrhosis",
            task_type="cirrhosis",
            input_data=data,
            modality="clinical_tabular",
        )

    # -------------------------------------------------------------------------

    def run_tumor_classification(
        self,
        image,
    ):
        """Run Tumor Classification Agent."""

        return self._execute_agent(
            agent_id="tumor_classification",
            task_type="tumor_classification",
            input_data=image,
            modality="2d_medical_image",
        )

    # -------------------------------------------------------------------------

    def run_liver_segmentation(
        self,
        volume,
    ):
        """Run Liver Segmentation Agent."""

        return self._execute_agent(
            agent_id="liver_segmentation",
            task_type="liver_segmentation",
            input_data=volume,
            modality="3d_medical_volume",
        )

    # =========================================================================
    # SPECIALIZED AGENTS
    # =========================================================================

    def run_specialized_agents(
        self,
        inputs,
    ):
        """
        Execute the five specialized diagnostic agents.

        Clinical reasoning is intentionally executed later because it should
        receive the outputs of the specialized agents.
        """

        results = {}

        # ---------------------------------------------------------------------
        # Fatty liver
        # ---------------------------------------------------------------------

        results["fatty_liver"] = self.run_fatty_liver(
            inputs.get("fatty_liver")
        )

        # ---------------------------------------------------------------------
        # Fibrosis
        # ---------------------------------------------------------------------

        results["fibrosis"] = self.run_fibrosis(
            inputs.get("fibrosis")
        )

        # ---------------------------------------------------------------------
        # Cirrhosis
        # ---------------------------------------------------------------------

        results["cirrhosis"] = self.run_cirrhosis(
            inputs.get("cirrhosis")
        )

        # ---------------------------------------------------------------------
        # Tumor
        # ---------------------------------------------------------------------

        results["tumor_classification"] = (
            self.run_tumor_classification(
                inputs.get("tumor")
            )
        )

        # ---------------------------------------------------------------------
        # Segmentation
        # ---------------------------------------------------------------------

        results["liver_segmentation"] = (
            self.run_liver_segmentation(
                inputs.get("segmentation")
            )
        )

        return results

    # =========================================================================
    # CLINICAL REASONING
    # =========================================================================

    def _build_clinical_reasoning_input(
        self,
        inputs,
        specialized_results,
    ):
        """
        Build input for Clinical Reasoning Agent.

        The current Clinical Reasoning model is trained on the six BUPA
        features. Therefore those features remain the direct model input.

        Specialized-agent outputs are additionally attached as context for
        the unified system.
        """

        direct_data = inputs.get(
            "clinical_reasoning"
        )

        if direct_data is None:

            direct_data = inputs.get(
                "fatty_liver"
            )

        if direct_data is None:
            direct_data = {}

        if not isinstance(
            direct_data,
            dict
        ):
            direct_data = {}

        # Keep only expected model features.
        model_data = {}

        for feature in self.CLINICAL_FEATURES:

            if feature in direct_data:

                model_data[feature] = (
                    direct_data[feature]
                )

        # ---------------------------------------------------------------------
        # If some features are absent, try fatty liver data.
        # ---------------------------------------------------------------------

        fatty_data = inputs.get(
            "fatty_liver"
        )

        if isinstance(fatty_data, dict):

            for feature in self.CLINICAL_FEATURES:

                if (
                    feature not in model_data
                    and feature in fatty_data
                ):

                    model_data[feature] = (
                        fatty_data[feature]
                    )

        return model_data

    # -------------------------------------------------------------------------

    def _run_clinical_reasoning(
        self,
        inputs,
        specialized_results,
    ):
        """
        Execute Clinical Reasoning after specialized agents.

        Returns a standardized dictionary.
        """

        agent_id = "clinical_reasoning"

        clinical_input = (
            self._build_clinical_reasoning_input(
                inputs=inputs,
                specialized_results=specialized_results,
            )
        )

        if self.clinical_agent is None:

            return {
                "agent_id": agent_id,
                "task_type": "clinical_reasoning",
                "modality": "clinical_tabular",
                "prediction": None,
                "probability": None,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "latency_ms": 0.0,
                "missing_data_ratio": 1.0,
                "trust": 0.0,
                "status": "not_available",
                "details": {},
                "explanation": (
                    "Clinical Reasoning Agent is not available."
                ),
                "error": (
                    "Clinical Reasoning Agent failed to load."
                ),
            }

        if not clinical_input:

            return {
                "agent_id": agent_id,
                "task_type": "clinical_reasoning",
                "modality": "clinical_tabular",
                "prediction": None,
                "probability": None,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "latency_ms": 0.0,
                "missing_data_ratio": 1.0,
                "trust": 0.0,
                "status": "not_run",
                "details": {},
                "explanation": (
                    "No clinical reasoning input supplied."
                ),
                "error": None,
            }

        # ---------------------------------------------------------------------
        # Execute clinical agent
        # ---------------------------------------------------------------------

        start_time = time.perf_counter()

        try:

            raw_output = self._call_agent(
                agent=self.clinical_agent,
                input_data=clinical_input,
                agent_id=agent_id,
            )

            latency_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

            result = self._normalize_agent_output(
                agent_id=agent_id,
                task_type="clinical_reasoning",
                raw_output=raw_output,
                latency_ms=latency_ms,
                modality="clinical_tabular",
            )

            result["status"] = "success"

            # Add specialized outputs as context.
            result.setdefault(
                "details",
                {}
            )

            if not isinstance(
                result["details"],
                dict
            ):
                result["details"] = {}

            result["details"][
                "specialized_agent_outputs"
            ] = specialized_results

            # Compute trust.
            result["trust"] = self._compute_trust(
                agent_id=agent_id,
                result=result,
            )

            return result

        except Exception as exc:

            latency_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

            error_text = (
                f"{type(exc).__name__}: {exc}"
            )

            self._log(
                "ERROR",
                "Clinical Reasoning execution failed: "
                f"{error_text}",
            )

            return {
                "agent_id": agent_id,
                "task_type": "clinical_reasoning",
                "modality": "clinical_tabular",
                "prediction": None,
                "probability": None,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "latency_ms": latency_ms,
                "missing_data_ratio": 1.0,
                "trust": 0.0,
                "status": "error",
                "details": {},
                "explanation": (
                    "Clinical Reasoning execution failed."
                ),
                "error": error_text,
                "traceback": traceback.format_exc(),
            }

    # =========================================================================
    # ADAPTIVE FUSION
    # =========================================================================

    def _run_adaptive_fusion(
        self,
        specialized_results,
    ):
        """
        Run AdaptiveFusion.

        AdaptiveFusion expects dictionaries, not AgentResult objects.
        """

        successful = []

        for result in specialized_results.values():

            if not isinstance(
                result,
                dict
            ):
                continue

            if result.get(
                "status"
            ) != "success":
                continue

            successful.append(
                result
            )

        if not successful:

            return {
                "status": "no_successful_agents",
                "fused_results": {},
                "message": (
                    "No specialized agent produced "
                    "a successful result."
                ),
            }

        if self.adaptive_fusion is None:

            # Safe fallback: evidence collection.
            return {
                "status": "fallback",
                "fused_results": {
                    result.get(
                        "task_type",
                        result.get("agent_id")
                    ): result
                    for result in successful
                },
                "message": (
                    "AdaptiveFusion unavailable. "
                    "Successful agent outputs preserved."
                ),
            }

        try:

            fused = self.adaptive_fusion.fuse(
                successful
            )

            return {
                "status": "success",
                "fused_results": fused,
                "message": (
                    "Adaptive fusion completed."
                ),
            }

        except Exception as exc:

            self._log(
                "WARNING",
                "Adaptive fusion failed: "
                f"{type(exc).__name__}: {exc}",
            )

            return {
                "status": "fallback",
                "fused_results": {
                    result.get(
                        "task_type",
                        result.get("agent_id")
                    ): result
                    for result in successful
                },
                "message": (
                    "AdaptiveFusion failed; "
                    "raw successful results preserved."
                ),
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    # =========================================================================
    # CONFLICT DETECTION
    # =========================================================================

    def _run_conflict_detection(
        self,
        specialized_results,
        clinical_result,
    ):
        """
        Detect conflicts between successful AgentResult objects.

        Note:
        The current ConflictDetector operates mainly by task type.
        Since the specialized agents have different task types, the detector
        may report no conflicts even when modalities contain clinically
        different evidence. This is expected behavior of the current module.
        """

        if self.conflict_detector is None:

            return {
                "status": "unavailable",
                "conflicts": [],
                "count": 0,
            }

        agent_results = []

        # ---------------------------------------------------------------------
        # Specialized results
        # ---------------------------------------------------------------------

        for result in specialized_results.values():

            try:

                if result.get(
                    "status"
                ) != "success":
                    continue

                agent_results.append(
                    self._to_agent_result(
                        result
                    )
                )

            except Exception:
                continue

        # ---------------------------------------------------------------------
        # Clinical reasoning
        # ---------------------------------------------------------------------

        if isinstance(
            clinical_result,
            dict
        ):

            if clinical_result.get(
                "status"
            ) == "success":

                try:

                    agent_results.append(
                        self._to_agent_result(
                            clinical_result
                        )
                    )

                except Exception:
                    pass

        # ---------------------------------------------------------------------
        # Nothing to compare
        # ---------------------------------------------------------------------

        if not agent_results:

            return {
                "status": "no_results",
                "conflicts": [],
                "count": 0,
            }

        try:

            conflicts = (
                self.conflict_detector.detect(
                    agent_results
                )
            )

            if conflicts is None:
                conflicts = []

            return {
                "status": "success",
                "conflicts": conflicts,
                "count": len(conflicts)
                if hasattr(conflicts, "__len__")
                else 0,
            }

        except Exception as exc:

            self._log(
                "WARNING",
                "Conflict detection failed: "
                f"{type(exc).__name__}: {exc}",
            )

            return {
                "status": "error",
                "conflicts": [],
                "count": 0,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    # =========================================================================
    # DECISION ENGINE
    # =========================================================================

    def _run_decision_engine(
        self,
        all_results,
        conflicts,
        clinical_result,
    ):
        """
        Produce final system-level decision.

        IMPORTANT:
        DecisionEngine.decide() expects:

            decide(results, conflicts=None, reasoning=None)

        It does NOT expect the old keyword arguments
        agent_results / fused_results / clinical_reasoning.
        """

        # ---------------------------------------------------------------------
        # Fallback
        # ---------------------------------------------------------------------

        if self.decision_engine is None:

            successful = [
                result
                for result in all_results
                if isinstance(result, dict)
                and result.get("status") == "success"
            ]

            confidences = [
                self._safe_probability(
                    result.get("confidence")
                )
                for result in successful
            ]

            confidences = [
                value
                for value in confidences
                if value is not None
            ]

            mean_confidence = (
                sum(confidences)
                / len(confidences)
                if confidences
                else 0.0
            )

            return {
                "status": "fallback",
                "prediction": (
                    clinical_result.get("prediction")
                    if isinstance(
                        clinical_result,
                        dict
                    )
                    else None
                ),
                "confidence": mean_confidence,
                "message": (
                    "DecisionEngine unavailable. "
                    "Fallback decision generated."
                ),
            }

        try:

            decision = self.decision_engine.decide(
                all_results,
                conflicts,
                clinical_result,
            )

            return {
                "status": "success",
                "decision": decision,
            }

        except Exception as exc:

            self._log(
                "WARNING",
                "Decision engine failed: "
                f"{type(exc).__name__}: {exc}",
            )

            # Safe fallback.
            prediction = None

            if isinstance(
                clinical_result,
                dict
            ):
                prediction = clinical_result.get(
                    "prediction"
                )

            return {
                "status": "fallback",
                "prediction": prediction,
                "confidence": 0.0,
                "message": (
                    "DecisionEngine failed. "
                    "Clinical reasoning output preserved."
                ),
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    # =========================================================================
    # PUBLIC RUN METHOD
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
        Run the complete LiverAI pipeline.

        Parameters
        ----------
        patient_id : str
            Unique patient identifier.

        patient_data : dict, optional
            Preferred unified input format.

        clinical_data : dict, optional
            Legacy / direct fatty-liver input.

        fibrosis_input : dict, optional
            Direct fibrosis input.

        cirrhosis_input : dict, optional
            Direct cirrhosis input.

        image : array/PIL image/tensor, optional
            2D tumor image.

        volume : array/tensor, optional
            3D liver volume.

        clinical_reasoning_input : dict, optional
            Direct clinical reasoning input.

        Returns
        -------
        dict
            Unified LiverAI report.
        """

        started_at = datetime.now()
        total_start = time.perf_counter()

        # =========================================================================
        # 1. NORMALIZE INPUT
        # =========================================================================

        try:

            inputs = self._extract_inputs(
                patient_data=patient_data,
                clinical_data=clinical_data,
                fibrosis_input=fibrosis_input,
                cirrhosis_input=cirrhosis_input,
                image=image,
                volume=volume,
                clinical_reasoning_input=(
                    clinical_reasoning_input
                ),
            )

        except Exception as exc:

            return {
                "patient_id": patient_id,
                "status": "error",
                "error": (
                    f"Input normalization failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "timestamp": started_at.isoformat(),
            }

        # =========================================================================
        # 2. EXECUTE SPECIALIZED AGENTS
        # =========================================================================

        self._log(
            "INFO",
            f"Starting patient analysis: {patient_id}",
        )

        specialized_results = (
            self.run_specialized_agents(
                inputs
            )
        )

        # =========================================================================
        # 3. ADAPTIVE FUSION
        # =========================================================================

        fusion_result = (
            self._run_adaptive_fusion(
                specialized_results
            )
        )

        # =========================================================================
        # 4. CONFLICT DETECTION
        # =========================================================================

        # Clinical reasoning has not yet been executed.
        # First detect conflicts among specialized agents.
        preliminary_conflict_result = (
            self._run_conflict_detection(
                specialized_results=
                    specialized_results,
                clinical_result=None,
            )
        )

        # =========================================================================
        # 5. CLINICAL REASONING
        # =========================================================================

        clinical_result = (
            self._run_clinical_reasoning(
                inputs=inputs,
                specialized_results=specialized_results,
            )
        )

        # =========================================================================
        # 6. FINAL CONFLICT DETECTION INCLUDING REASONING
        # =========================================================================

        conflict_result = (
            self._run_conflict_detection(
                specialized_results=
                    specialized_results,
                clinical_result=clinical_result,
            )
        )

        # =========================================================================
        # 7. COLLECT ALL RESULTS
        # =========================================================================

        all_results = []

        for result in specialized_results.values():

            if isinstance(
                result,
                dict
            ):
                all_results.append(
                    result
                )

        if isinstance(
            clinical_result,
            dict
        ):
            all_results.append(
                clinical_result
            )

        # =========================================================================
        # 8. FINAL DECISION
        # =========================================================================

        conflicts = conflict_result.get(
            "conflicts",
            []
        )

        decision_result = (
            self._run_decision_engine(
                all_results=all_results,
                conflicts=conflicts,
                clinical_result=clinical_result,
            )
        )

        # =========================================================================
        # 9. EXECUTION STATISTICS
        # =========================================================================

        total_latency_ms = (
            time.perf_counter()
            - total_start
        ) * 1000.0

        successful_agents = [
            result
            for result in all_results
            if result.get("status") == "success"
        ]

        failed_agents = [
            result
            for result in all_results
            if result.get("status") == "error"
        ]

        not_run_agents = [
            result
            for result in all_results
            if result.get("status") in (
                "not_run",
                "not_available",
            )
        ]

        # =========================================================================
        # 10. SYSTEM STATUS
        # =========================================================================

        if len(successful_agents) == 6:

            system_status = "complete"

        elif len(successful_agents) > 0:

            system_status = "partial"

        else:

            system_status = "failed"

        # =========================================================================
        # 11. INPUT AVAILABILITY
        # =========================================================================

        input_availability = {

            "fatty_liver": self._has_value(
                inputs.get("fatty_liver")
            ),

            "fibrosis": self._has_value(
                inputs.get("fibrosis")
            ),

            "cirrhosis": self._has_value(
                inputs.get("cirrhosis")
            ),

            "tumor": self._has_value(
                inputs.get("tumor")
            ),

            "segmentation": self._has_value(
                inputs.get("segmentation")
            ),

            "clinical_reasoning": self._has_value(
                inputs.get("clinical_reasoning")
            ),
        }

        # =========================================================================
        # 12. FINAL UNIFIED REPORT
        # =========================================================================

        report = {

            "patient_id": patient_id,

            "timestamp": started_at.isoformat(),

            "status": system_status,

            # -----------------------------------------------------------------
            # Agent outputs
            # -----------------------------------------------------------------

            "agents": {

                "fatty_liver":
                    specialized_results.get(
                        "fatty_liver"
                    ),

                "fibrosis":
                    specialized_results.get(
                        "fibrosis"
                    ),

                "cirrhosis":
                    specialized_results.get(
                        "cirrhosis"
                    ),

                "tumor_classification":
                    specialized_results.get(
                        "tumor_classification"
                    ),

                "liver_segmentation":
                    specialized_results.get(
                        "liver_segmentation"
                    ),

                "clinical_reasoning":
                    clinical_result,
            },

            # -----------------------------------------------------------------
            # Coordination
            # -----------------------------------------------------------------

            "coordination": {

                "adaptive_fusion":
                    fusion_result,

                "conflicts":
                    conflict_result,

                "preliminary_conflicts":
                    preliminary_conflict_result,

                "decision":
                    decision_result,
            },

            # -----------------------------------------------------------------
            # Direct aliases
            # -----------------------------------------------------------------

            "adaptive_fusion":
                fusion_result,

            "conflicts":
                conflict_result,

            "clinical_reasoning":
                clinical_result,

            "decision":
                decision_result,

            # -----------------------------------------------------------------
            # Execution
            # -----------------------------------------------------------------

            "execution": {

                "total_latency_ms":
                    total_latency_ms,

                "successful_agents":
                    len(successful_agents),

                "failed_agents":
                    len(failed_agents),

                "not_run_agents":
                    len(not_run_agents),

                "total_agents":
                    6,

                "input_availability":
                    input_availability,
            },

            # -----------------------------------------------------------------
            # Metadata
            # -----------------------------------------------------------------

            "metadata": {

                "orchestrator":
                    self.name,

                "architecture":
                    "multimodal_multi_agent",

                "agent_count":
                    6,

                "specialized_agent_count":
                    5,

                "clinical_reasoning_enabled":
                    self.clinical_agent is not None,

                "trust_manager_enabled":
                    self.trust_manager is not None,

                "adaptive_fusion_enabled":
                    self.adaptive_fusion is not None,

                "conflict_detection_enabled":
                    self.conflict_detector is not None,

                "decision_engine_enabled":
                    self.decision_engine is not None,
            },

            # -----------------------------------------------------------------
            # Safety
            # -----------------------------------------------------------------

            "clinical_note": (
                "This system output is a computational decision-support "
                "result and must not be considered a standalone medical "
                "diagnosis. Clinical interpretation requires qualified "
                "medical review."
            ),
        }

        # =========================================================================
        # 13. FINAL LOG
        # =========================================================================

        self._log(
            "INFO",
            (
                f"Patient {patient_id} completed: "
                f"{len(successful_agents)}/6 agents successful "
                f"in {total_latency_ms:.2f} ms"
            ),
        )

        return report

    # =========================================================================
    # ALIAS
    # =========================================================================

    def analyze(
        self,
        patient_id,
        patient_data=None,
        **kwargs,
    ):
        """
        Alias for run().

        Allows:

            orchestrator.analyze(
                patient_id="PATIENT_001",
                patient_data=data
            )
        """

        return self.run(
            patient_id=patient_id,
            patient_data=patient_data,
            **kwargs,
        )

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self):
        """
        Return the status of all agents and coordination modules.
        """

        agent_status = {}

        for agent_id, agent in self.agents.items():

            agent_status[agent_id] = {
                "loaded": agent is not None
            }

        coordination_status = {

            "trust_manager":
                self.trust_manager is not None,

            "adaptive_fusion":
                self.adaptive_fusion is not None,

            "conflict_detector":
                self.conflict_detector is not None,

            "decision_engine":
                self.decision_engine is not None,
        }

        loaded_agents = sum(
            1
            for agent in self.agents.values()
            if agent is not None
        )

        return {

            "status": (
                "healthy"
                if loaded_agents == 6
                else "partial"
            ),

            "agents_loaded":
                loaded_agents,

            "agents_total":
                6,

            "agents":
                agent_status,

            "coordination":
                coordination_status,
        }


# =============================================================================
# END OF FILE
# =============================================================================
