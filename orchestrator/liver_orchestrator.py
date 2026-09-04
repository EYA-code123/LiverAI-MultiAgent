# =============================================================================
# LiverAI-MultiAgent
# COMPLETE MULTI-MODAL MULTI-AGENT ORCHESTRATOR
# =============================================================================

import traceback
from datetime import datetime


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
# ORCHESTRATOR
# =============================================================================

class LiverAIOrchestrator:
    """
    Main LiverAI multi-agent orchestration layer.

    Agents
    ------
    1. Fatty Liver Classification
    2. Fibrosis Prediction
    3. Cirrhosis Classification
    4. Tumor Classification
    5. Liver Segmentation
    6. Clinical Reasoning

    Important
    ---------
    Each agent receives ONLY the input modality/features required by
    its own model.

    Fatty Liver:
        mcv, alkphos, sgpt, sgot, gammagt, drinks

    Fibrosis:
        age, male, weight, height, bmi, futime, days, test, value

    Cirrhosis:
        18 specific clinical features

    Tumor:
        2D MRI image

    Segmentation:
        3D liver volume

    Clinical Reasoning:
        mcv, alkphos, sgpt, sgot, gammagt, drinks
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self):

        self.name = "LiverAI Multi-Agent Orchestrator"

        # ---------------------------------------------------------------------
        # MODEL PATHS
        # ---------------------------------------------------------------------

        self.model_paths = {

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

        # ---------------------------------------------------------------------
        # AGENT OBJECTS
        # ---------------------------------------------------------------------

        self.fatty_liver_agent = None
        self.fibrosis_agent = None
        self.cirrhosis_agent = None
        self.tumor_agent = None
        self.segmentation_agent = None
        self.clinical_agent = None

        # ---------------------------------------------------------------------
        # LOAD ALL AGENTS
        # ---------------------------------------------------------------------

        self._load_fatty_liver_agent()
        self._load_fibrosis_agent()
        self._load_cirrhosis_agent()
        self._load_tumor_agent()
        self._load_segmentation_agent()
        self._load_clinical_reasoning_agent()

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
        # STATE
        # ---------------------------------------------------------------------

        self.last_results = {}
        self.last_assessment = None
        self.execution_log = []

        # ---------------------------------------------------------------------
        # SYSTEM STATUS
        # ---------------------------------------------------------------------

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
                f"  {name:<25} : {status}"
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
            f"  Conflict Detector      : "
            f"{'READY' if self.conflict_detector else 'FALLBACK'}"
        )

        print(
            f"  Decision Engine        : "
            f"{'READY' if self.decision_engine else 'FALLBACK'}"
        )

        print("=" * 80)

    # =========================================================================
    # AGENT LOADING
    # =========================================================================

    def _load_fatty_liver_agent(self):

        try:

            import joblib
            from agents.fatty_liver_agent import FattyLiverAgent

            model = joblib.load(
                self.model_paths["fatty_liver"]
            )

            self.fatty_liver_agent = FattyLiverAgent(
                model
            )

            print("✓ Fatty Liver Agent loaded")

        except Exception as e:

            print("✗ Fatty Liver Agent failed:")
            print(
                f"  {type(e).__name__}: {e}"
            )

            self.fatty_liver_agent = None

    # -------------------------------------------------------------------------

    def _load_fibrosis_agent(self):

        try:

            import joblib
            from agents.fibrosis_agent import FibrosisAgent

            model = joblib.load(
                self.model_paths["fibrosis"]
            )

            self.fibrosis_agent = FibrosisAgent(
                model
            )

            print("✓ Fibrosis Agent loaded")

        except Exception as e:

            print("✗ Fibrosis Agent failed:")
            print(
                f"  {type(e).__name__}: {e}"
            )

            self.fibrosis_agent = None

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
            print(
                f"  {type(e).__name__}: {e}"
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
                "✗ Tumor Classification Agent failed:"
            )

            print(
                f"  {type(e).__name__}: {e}"
            )

            self.tumor_agent = None

    # -------------------------------------------------------------------------

    def _load_segmentation_agent(self):

        try:

            from agents.liver_segmentation_agent import (
                LiverSegmentationAgent
            )

            self.segmentation_agent = LiverSegmentationAgent(
                model_path=
                    self.model_paths[
                        "liver_segmentation"
                    ]
            )

            print(
                "✓ Liver Segmentation Agent loaded"
            )

        except Exception as e:

            print(
                "✗ Liver Segmentation Agent failed:"
            )

            print(
                f"  {type(e).__name__}: {e}"
            )

            self.segmentation_agent = None

    # -------------------------------------------------------------------------

    def _load_clinical_reasoning_agent(self):

        try:

            from agents.clinical_reasoning_agent import (
                ClinicalReasoningAgent
            )

            self.clinical_agent = ClinicalReasoningAgent(
                self.model_paths[
                    "clinical_reasoning"
                ]
            )

            print(
                "✓ Clinical Reasoning Agent loaded"
            )

        except Exception as e:

            print(
                "✗ Clinical Reasoning Agent failed:"
            )

            print(
                f"  {type(e).__name__}: {e}"
            )

            self.clinical_agent = None

    # =========================================================================
    # LOGGING
    # =========================================================================

    def _log(
        self,
        message
    ):

        timestamp = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        self.execution_log.append(
            {
                "timestamp": timestamp,
                "message": message,
            }
        )

        print(message)

    # =========================================================================
    # EMPTY RESULT
    # =========================================================================

    def _empty_result(
        self,
        agent_name,
        task_type,
        status="not_available",
        error=None,
    ):

        return {

            "agent_id":
                agent_name,

            "agent":
                agent_name,

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
        }

    # =========================================================================
    # SAFE AGENT EXECUTION
    # =========================================================================

    def _execute_agent(
        self,
        agent_name,
        agent,
        input_data,
        task_type,
    ):

        start_time = datetime.now()

        self._log(
            f"\n[{agent_name}] START"
        )

        # ---------------------------------------------------------------------
        # AGENT UNAVAILABLE
        # ---------------------------------------------------------------------

        if agent is None:

            self._log(
                f"[{agent_name}] NOT AVAILABLE"
            )

            return self._empty_result(
                agent_name,
                task_type,
                status="not_available",
                error="Agent not available.",
            )

        # ---------------------------------------------------------------------
        # INPUT UNAVAILABLE
        # ---------------------------------------------------------------------

        if input_data is None:

            self._log(
                f"[{agent_name}] NO INPUT → SKIPPED"
            )

            return self._empty_result(
                agent_name,
                task_type,
                status="not_available",
                error="Required input not provided.",
            )

        # ---------------------------------------------------------------------
        # EXECUTE
        # ---------------------------------------------------------------------

        try:

            if hasattr(
                agent,
                "predict"
            ):

                result = agent.predict(
                    input_data
                )

            elif hasattr(
                agent,
                "run"
            ):

                result = agent.run(
                    input_data
                )

            elif hasattr(
                agent,
                "analyze"
            ):

                result = agent.analyze(
                    input_data
                )

            else:

                raise AttributeError(
                    f"{agent_name} has no "
                    "predict(), run(), or analyze() method."
                )

            # -----------------------------------------------------------------
            # NORMALIZE RESULT
            # -----------------------------------------------------------------

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
                datetime.now()
                -
                start_time
            ).total_seconds() * 1000.0

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
            # NORMALIZE NUMERIC VALUES
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

                result["latency_ms"] = float(
                    max(
                        0.0,
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
                f"| status={result['status']} "
                f"| confidence="
                f"{result['confidence']:.3f}"
            )

            return result

        # ---------------------------------------------------------------------
        # ERROR
        # ---------------------------------------------------------------------

        except Exception as e:

            elapsed_ms = (
                datetime.now()
                -
                start_time
            ).total_seconds() * 1000.0

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
    # INPUT EXTRACTION
    # =========================================================================

    @staticmethod
    def _extract_input(
        data,
        key,
    ):
        """
        Extract modality-specific input.

        Supports:

            {
                "fatty_liver": {...},
                "fibrosis": {...},
                "cirrhosis": {...},
                "tumor": image,
                "segmentation": volume,
                "clinical_reasoning": {...}
            }

        Also supports a direct input when `data` itself is the
        modality-specific object.
        """

        if data is None:

            return None

        if isinstance(
            data,
            dict
        ):

            if key in data:

                return data[key]

        return data

    # =========================================================================
    # INDIVIDUAL AGENTS
    # =========================================================================

    def run_fatty_liver(
        self,
        clinical_data
    ):

        return self._execute_agent(
            "FattyLiverAgent",
            self.fatty_liver_agent,
            clinical_data,
            "fatty_liver_classification",
        )

    # -------------------------------------------------------------------------

    def run_fibrosis(
        self,
        fibrosis_input
    ):

        return self._execute_agent(
            "FibrosisAgent",
            self.fibrosis_agent,
            fibrosis_input,
            "fibrosis_prediction",
        )

    # -------------------------------------------------------------------------

    def run_cirrhosis(
        self,
        cirrhosis_input
    ):

        return self._execute_agent(
            "CirrhosisAgent",
            self.cirrhosis_agent,
            cirrhosis_input,
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
            "RUNNING SPECIALIZED AGENTS"
        )

        self._log(
            "=" * 80
        )

        # ---------------------------------------------------------------------
        # SUPPORT TWO INPUT STYLES
        # ---------------------------------------------------------------------
        #
        # New recommended style:
        #
        # patient_data = {
        #     "fatty_liver": {...},
        #     "fibrosis": {...},
        #     "cirrhosis": {...},
        #     "tumor": image,
        #     "segmentation": volume,
        #     "clinical_reasoning": {...}
        # }
        #
        # Legacy style:
        #
        # clinical_data=...
        # fibrosis_input=...
        # cirrhosis_input=...
        # image=...
        # volume=...
        # ---------------------------------------------------------------------

        if patient_data is not None:

            fatty_input = self._extract_input(
                patient_data,
                "fatty_liver"
            )

            fibrosis_data = self._extract_input(
                patient_data,
                "fibrosis"
            )

            cirrhosis_data = self._extract_input(
                patient_data,
                "cirrhosis"
            )

            tumor_data = self._extract_input(
                patient_data,
                "tumor"
            )

            segmentation_data = self._extract_input(
                patient_data,
                "segmentation"
            )

        else:

            fatty_input = clinical_data

            fibrosis_data = fibrosis_input

            cirrhosis_data = cirrhosis_input

            tumor_data = image

            segmentation_data = volume

        # ---------------------------------------------------------------------
        # 1. FATTY LIVER
        # ---------------------------------------------------------------------

        self._log(
            "\n--- 1/5 Fatty Liver ---"
        )

        fatty_result = self.run_fatty_liver(
            fatty_input
        )

        # ---------------------------------------------------------------------
        # 2. FIBROSIS
        # ---------------------------------------------------------------------

        self._log(
            "\n--- 2/5 Fibrosis ---"
        )

        fibrosis_result = self.run_fibrosis(
            fibrosis_data
        )

        # ---------------------------------------------------------------------
        # 3. CIRRHOSIS
        # ---------------------------------------------------------------------

        self._log(
            "\n--- 3/5 Cirrhosis ---"
        )

        cirrhosis_result = self.run_cirrhosis(
            cirrhosis_data
        )

        # ---------------------------------------------------------------------
        # 4. TUMOR
        # ---------------------------------------------------------------------

        self._log(
            "\n--- 4/5 Tumor Classification ---"
        )

        tumor_result = self.run_tumor_classification(
            tumor_data
        )

        # ---------------------------------------------------------------------
        # 5. SEGMENTATION
        # ---------------------------------------------------------------------

        self._log(
            "\n--- 5/5 Liver Segmentation ---"
        )

        segmentation_result = self.run_liver_segmentation(
            segmentation_data
        )

        # ---------------------------------------------------------------------
        # RESULTS
        # ---------------------------------------------------------------------

        results = {

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

        self.last_results = results

        return results

    # =========================================================================
    # STANDARDIZATION + TRUST
    # =========================================================================

    def _to_agent_results(
        self,
        raw_results
    ):

        agent_results = []

        # ---------------------------------------------------------------------
        # FALLBACK WHEN AgentResult IS UNAVAILABLE
        # ---------------------------------------------------------------------

        if AgentResult is None:

            return raw_results

        # ---------------------------------------------------------------------
        # CONVERT
        # ---------------------------------------------------------------------

        for result in raw_results.values():

            try:

                agent_result = (
                    AgentResult.from_dict(
                        result
                    )
                )

            except Exception as e:

                self._log(
                    "AgentResult conversion failed: "
                    f"{e}"
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

                    agent_result.trust = (
                        self._clip(trust)
                    )

                except Exception as e:

                    self._log(
                        f"Trust computation failed: {e}"
                    )

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
    # CLINICAL REASONING
    # =========================================================================

    def run_clinical_reasoning(
        self,
        clinical_input,
        agent_results=None,
        fusion_result=None,
        conflicts=None,
    ):

        self._log(
            "\n" + "=" * 80
        )

        self._log(
            "CLINICAL REASONING AGENT"
        )

        self._log(
            "=" * 80
        )

        # ---------------------------------------------------------------------
        # AGENT UNAVAILABLE
        # ---------------------------------------------------------------------

        if self.clinical_agent is None:

            return self._empty_result(
                "ClinicalReasoningAgent",
                "clinical_reasoning",
                status="not_available",
                error=(
                    "Clinical reasoning agent "
                    "not available."
                ),
            )

        # ---------------------------------------------------------------------
        # INPUT UNAVAILABLE
        # ---------------------------------------------------------------------

        if clinical_input is None:

            return self._empty_result(
                "ClinicalReasoningAgent",
                "clinical_reasoning",
                status="not_available",
                error=(
                    "Clinical reasoning input "
                    "not provided."
                ),
            )

        # ---------------------------------------------------------------------
        # IMPORTANT
        # ---------------------------------------------------------------------
        #
        # The ClinicalReasoningAgent expects the six BUPA features:
        #
        # mcv
        # alkphos
        # sgpt
        # sgot
        # gammagt
        # drinks
        #
        # We DO NOT send:
        #
        # {
        #     "agents": ...,
        #     "fusion": ...,
        #     "conflicts": ...
        # }
        #
        # as the model input.
        # ---------------------------------------------------------------------

        try:

            start_time = datetime.now()

            result = self.clinical_agent.predict(
                clinical_input
            )

            elapsed_ms = (
                datetime.now()
                -
                start_time
            ).total_seconds() * 1000.0

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
                "prediction",
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

            self._log(
                "✓ Clinical reasoning completed "
                f"| confidence="
                f"{result['confidence']:.3f}"
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

                "details":
                    {},

                "error":
                    str(e),

                "traceback":
                    traceback.format_exc(),
            }

    # =========================================================================
    # FALLBACK FUSION
    # =========================================================================

    def _fallback_fusion(
        self,
        agent_results
    ):

        evidence = []

        for result in agent_results:

            if result.status not in [
                "success",
                "completed",
            ]:

                continue

            if result.prediction is None:

                continue

            evidence.append(
                {

                    "agent_id":
                        result.agent_id,

                    "task_type":
                        result.task_type,

                    "prediction":
                        result.prediction,

                    "probability":
                        getattr(
                            result,
                            "probability",
                            None
                        ),

                    "confidence":
                        result.confidence,

                    "uncertainty":
                        result.uncertainty,

                    "quality":
                        result.quality,

                    "trust":
                        getattr(
                            result,
                            "trust",
                            None
                        ),
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
                        (
                            item["trust"]
                            if item["trust"] is not None
                            else item["confidence"]
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

        # ---------------------------------------------------------------------
        # GROUP BY TASK
        # ---------------------------------------------------------------------

        task_groups = {}

        for result in agent_results:

            task_groups.setdefault(
                result.task_type,
                []
            ).append(
                result
            )

        # ---------------------------------------------------------------------
        # SAME-TASK CONFLICTS ONLY
        # ---------------------------------------------------------------------

        for task_type, group in task_groups.items():

            predictions = [

                str(
                    item.prediction
                )

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
        clinical_reasoning
    ):

        valid = [

            result

            for result in agent_results

            if result.status in [
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
                    (
                        "Insufficient valid agent "
                        "outputs. This is not a "
                        "medical diagnosis."
                    ),
            }

        # ---------------------------------------------------------------------
        # AVERAGE CONFIDENCE
        # ---------------------------------------------------------------------

        confidences = [

            self._clip(
                result.confidence
            )

            for result in valid
        ]

        average_confidence = (

            sum(confidences)
            /
            len(confidences)

        )

        # ---------------------------------------------------------------------
        # SYSTEM-LEVEL EVIDENCE SCORE
        # ---------------------------------------------------------------------

        risk_components = []

        for result in valid:

            prediction = str(
                result.prediction
            ).lower()

            # -------------------------------------------------------------
            # Only system-level classification keywords.
            # -------------------------------------------------------------

            risk_keywords = [

                "tumor",
                "carcinoma",
                "angiosarcoma",
                "cholangiocarcinoma",
                "fibrosis",
                "cirrhosis",
                "positive",
                "abnormal",
            ]

            if any(
                word in prediction
                for word in risk_keywords
            ):

                risk_components.append(
                    self._clip(
                        result.confidence
                    )
                )

        # ---------------------------------------------------------------------
        # RISK SCORE
        # ---------------------------------------------------------------------

        risk_score = (

            sum(risk_components)
            /
            len(risk_components)

            if risk_components

            else 0.0
        )

        # ---------------------------------------------------------------------
        # RISK LEVEL
        # ---------------------------------------------------------------------

        if risk_score >= 0.70:

            risk_level = "high"

        elif risk_score >= 0.40:

            risk_level = "moderate"

        else:

            risk_level = "low"

        # ---------------------------------------------------------------------
        # CLINICAL REASONING INFORMATION
        # ---------------------------------------------------------------------

        clinical_prediction = None

        clinical_confidence = 0.0

        if isinstance(
            clinical_reasoning,
            dict
        ):

            clinical_prediction = (
                clinical_reasoning.get(
                    "prediction"
                )
            )

            clinical_confidence = (
                self._clip(
                    clinical_reasoning.get(
                        "confidence",
                        0.0
                    )
                )
            )

        return {

            "status":
                "completed",

            "risk_level":
                risk_level,

            "risk_score":
                float(
                    risk_score
                ),

            "decision_confidence":
                float(
                    average_confidence
                ),

            "clinical_reasoning_prediction":
                clinical_prediction,

            "clinical_reasoning_confidence":
                clinical_confidence,

            "request_additional_tests":
                bool(
                    len(conflicts) > 0
                ),

            "conflicts_detected":
                len(conflicts),

            "note":
                (
                    "This is a system-level evidence "
                    "aggregation and not a medical diagnosis."
                ),
        }

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

        # ---------------------------------------------------------------------
        # RESET LOG
        # ---------------------------------------------------------------------

        self.execution_log = []

        self._log(
            "\n" + "=" * 80
        )

        self._log(
            f"LIVERAI ANALYSIS — PATIENT {patient_id}"
        )

        self._log(
            "=" * 80
        )

        # =========================================================================
        # STEP 1 — SPECIALIZED AGENTS
        # =========================================================================

        self._log(
            "\nSTEP 1/6 → Specialized Agents"
        )

        raw_results = (
            self.run_specialized_agents(

                patient_data=
                    patient_data,

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
        # STEP 2 — STANDARDIZATION + TRUST
        # =========================================================================

        self._log(
            "\nSTEP 2/6 → Standardization + Trust"
        )

        agent_results = (
            self._to_agent_results(
                raw_results
            )
        )

        # ---------------------------------------------------------------------
        # SAFETY CHECK
        # ---------------------------------------------------------------------

        if AgentResult is None:

            self._log(
                "AgentResult unavailable → "
                "using raw result objects."
            )

        # =========================================================================
        # STEP 3 — ADAPTIVE FUSION
        # =========================================================================

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

        # =========================================================================
        # STEP 4 — CONFLICT DETECTION
        # =========================================================================

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

                if conflicts is None:

                    conflicts = []

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
        # STEP 5 — CLINICAL REASONING
        # =========================================================================

        self._log(
            "\nSTEP 5/6 → Clinical Reasoning"
        )

        # ---------------------------------------------------------------------
        # GET CLINICAL INPUT
        # ---------------------------------------------------------------------

        if clinical_reasoning_input is not None:

            clinical_input = (
                clinical_reasoning_input
            )

        elif patient_data is not None:

            clinical_input = self._extract_input(
                patient_data,
                "clinical_reasoning"
            )

        else:

            clinical_input = clinical_data

        # ---------------------------------------------------------------------
        # RUN CLINICAL REASONING
        # ---------------------------------------------------------------------

        clinical_result = (
            self.run_clinical_reasoning(

                clinical_input=

                    clinical_input,

                agent_results=
                    agent_results,

                fusion_result=
                    fusion_result,

                conflicts=
                    conflicts,
            )
        )

        # =========================================================================
        # STEP 6 — DECISION ENGINE
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

        # ---------------------------------------------------------------------
        # SPECIALIZED AGENTS
        # ---------------------------------------------------------------------

        if AgentResult is not None:

            for result in agent_results:

                try:

                    serialized_agents[
                        result.agent_id
                    ] = result.to_dict()

                except Exception:

                    serialized_agents[
                        result.agent_id
                    ] = {
                        "agent_id":
                            result.agent_id,

                        "prediction":
                            getattr(
                                result,
                                "prediction",
                                None
                            ),

                        "confidence":
                            getattr(
                                result,
                                "confidence",
                                0.0
                            ),

                        "status":
                            getattr(
                                result,
                                "status",
                                "unknown"
                            ),
                    }

        else:

            serialized_agents = raw_results.copy()

        # ---------------------------------------------------------------------
        # CLINICAL REASONING
        # ---------------------------------------------------------------------

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

            # ---------------------------------------------------------------
            # AGENT COUNTS
            # ---------------------------------------------------------------

            "total_specialized_agents":
                5,

            "total_agents":
                6,

            "agents_completed":
                len(
                    completed_agents
                ),

            "completed_agent_names":
                completed_agents,

            # ---------------------------------------------------------------
            # RESULTS
            # ---------------------------------------------------------------

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

            # ---------------------------------------------------------------
            # EXECUTION
            # ---------------------------------------------------------------

            "execution_log":
                self.execution_log,

            # ---------------------------------------------------------------
            # SYSTEM NOTE
            # ---------------------------------------------------------------

            "note":
                (
                    "LiverAI provides model-based evidence "
                    "aggregation for research/engineering "
                    "purposes. Outputs are not medical diagnoses."
                ),
        }

        # =========================================================================
        # SAVE STATE
        # =========================================================================

        self.last_results = final_result

        self.last_assessment = final_result

        # =========================================================================
        # END
        # =========================================================================

        self._log(
            "\n" + "=" * 80
        )

        self._log(
            "LIVERAI PIPELINE COMPLETED"
        )

        self._log(
            "=" * 80
        )

        return final_result

    # =========================================================================
    # SYSTEM STATUS
    # =========================================================================

    def get_system_status(
        self
    ):

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
    # LAST RESULTS
    # =========================================================================

    def get_last_results(
        self
    ):

        return self.last_results

    # =========================================================================

    def get_last_assessment(
        self
    ):

        return self.last_assessment

    # =========================================================================

    def get_execution_log(
        self
    ):

        return self.execution_log

    # =========================================================================
    # UTILITY
    # =========================================================================

    @staticmethod
    def _clip(
        value
    ):

        try:

            value = float(
                value
            )

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
