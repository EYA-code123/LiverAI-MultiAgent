# =============================================================================
# LiverAI-MultiAgent
# ADAPTIVE MULTI-AGENT ORCHESTRATOR
# =============================================================================

import traceback
from datetime import datetime

from orchestrator.schemas import AgentResult

from coordinator.trust import TrustManager
from coordinator.adaptive_fusion import AdaptiveFusion
from coordinator.conflict import ConflictDetector
from coordinator.decision import DecisionEngine


class LiverAIOrchestrator:
    """
    Main coordination layer of LiverAI.

    Pipeline:

        Patient Data
             |
             v
        Specialized Agents
             |
             v
        Standardization
             |
             v
        Adaptive Trust
             |
             v
        Adaptive Evidence Fusion
             |
             v
        Conflict Detection
             |
             v
        Clinical Reasoning
             |
             v
        Decision Intelligence
             |
             v
        Final Coordinated Result
    """

    def __init__(
        self,
        cirrhosis_agent=None,
        fatty_liver_agent=None,
        clinical_agent=None,
        fibrosis_agent=None,
        tumor_agent=None,
        segmentation_agent=None,
    ):

        self.name = (
            "LiverAI Adaptive Multi-Agent Orchestrator"
        )

        # ---------------------------------------------------------------------
        # SPECIALIZED AGENTS
        # ---------------------------------------------------------------------

        self.cirrhosis_agent = (
            cirrhosis_agent
        )

        self.fatty_liver_agent = (
            fatty_liver_agent
        )

        self.clinical_agent = (
            clinical_agent
        )

        self.fibrosis_agent = (
            fibrosis_agent
        )

        self.tumor_agent = (
            tumor_agent
        )

        self.segmentation_agent = (
            segmentation_agent
        )

        self.agents = {

            "cirrhosis":
                cirrhosis_agent,

            "fatty_liver":
                fatty_liver_agent,

            "fibrosis":
                fibrosis_agent,

            "tumor_classification":
                tumor_agent,

            "liver_segmentation":
                segmentation_agent,

            "clinical_reasoning":
                clinical_agent,
        }

        # ---------------------------------------------------------------------
        # COORDINATION MODULES
        # ---------------------------------------------------------------------

        self.trust_manager = TrustManager()

        self.adaptive_fusion = AdaptiveFusion()

        self.conflict_detector = (
            ConflictDetector()
        )

        self.decision_engine = (
            DecisionEngine()
        )

        # ---------------------------------------------------------------------
        # STATE
        # ---------------------------------------------------------------------

        self.last_results = {}

        self.execution_log = []

    # =========================================================================
    # LOGGING
    # =========================================================================

    def _log(self, message):

        timestamp = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        self.execution_log.append({

            "timestamp":
                timestamp,

            "message":
                message,
        })

        print(message)

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

        if agent is None:

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

                "latency_ms":
                    0.0,

                "missing_data_ratio":
                    1.0,

                "details":
                    {},

                "error":
                    "Agent not available",
            }

        if input_data is None:

            return {

                "agent_id":
                    agent_name,

                "agent":
                    agent_name,

                "task_type":
                    task_type,

                "status":
                    "no_input",

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

                "latency_ms":
                    0.0,

                "missing_data_ratio":
                    1.0,

                "details":
                    {},

                "error":
                    "No input provided",
            }

        self._log(
            f"\n[{agent_name}] START"
        )

        start_time = datetime.now()

        try:

            result = agent.predict(
                input_data
            )

            # -------------------------------------------------------------
            # NORMALIZE RETURN TYPE
            # -------------------------------------------------------------

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

            # -------------------------------------------------------------
            # BASIC FIELDS
            # -------------------------------------------------------------

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
                "success"
            )

            result.setdefault(
                "prediction",
                None
            )

            result.setdefault(
                "probability",
                None
            )

            # -------------------------------------------------------------
            # CONFIDENCE
            # -------------------------------------------------------------

            if result.get(
                "confidence"
            ) is None:

                probability = result.get(
                    "probability"
                )

                if isinstance(
                    probability,
                    (int, float)
                ):

                    result[
                        "confidence"
                    ] = float(
                        probability
                    )

                elif isinstance(
                    probability,
                    (list, tuple)
                ) and probability:

                    result[
                        "confidence"
                    ] = float(
                        max(
                            probability
                        )
                    )

                else:

                    result[
                        "confidence"
                    ] = 0.0

            # -------------------------------------------------------------
            # UNCERTAINTY
            # -------------------------------------------------------------

            result.setdefault(
                "uncertainty",
                1.0
                -
                float(
                    result.get(
                        "confidence",
                        0.0
                    )
                )
            )

            # -------------------------------------------------------------
            # QUALITY
            # -------------------------------------------------------------

            result.setdefault(
                "quality",
                1.0
            )

            # -------------------------------------------------------------
            # LATENCY
            # -------------------------------------------------------------

            elapsed = (
                datetime.now()
                -
                start_time
            ).total_seconds() * 1000.0

            result.setdefault(
                "latency_ms",
                elapsed
            )

            # -------------------------------------------------------------
            # MISSING DATA
            # -------------------------------------------------------------

            result.setdefault(
                "missing_data_ratio",
                0.0
            )

            # -------------------------------------------------------------
            # DETAILS
            # -------------------------------------------------------------

            result.setdefault(
                "details",
                {}
            )

            if not isinstance(
                result["details"],
                dict
            ):

                result["details"] = {}

            result[
                "details"
            ].setdefault(
                "task_type",
                task_type
            )

            # -------------------------------------------------------------
            # EXPLANATION
            # -------------------------------------------------------------

            result.setdefault(
                "explanation",
                None
            )

            # -------------------------------------------------------------
            # ERROR
            # -------------------------------------------------------------

            result.setdefault(
                "error",
                None
            )

            if result["error"]:

                result["status"] = (
                    "error"
                )

            # -------------------------------------------------------------
            # CLEAN NUMERIC VALUES
            # -------------------------------------------------------------

            result[
                "confidence"
            ] = self._clip(
                result.get(
                    "confidence",
                    0.0
                )
            )

            result[
                "uncertainty"
            ] = self._clip(
                result.get(
                    "uncertainty",
                    1.0
                )
            )

            result[
                "quality"
            ] = self._clip(
                result.get(
                    "quality",
                    0.0
                )
            )

            result[
                "missing_data_ratio"
            ] = self._clip(
                result.get(
                    "missing_data_ratio",
                    0.0
                )
            )

            result[
                "latency_ms"
            ] = float(
                max(
                    0.0,
                    result.get(
                        "latency_ms",
                        elapsed
                    )
                )
            )

            self._log(
                f"[{agent_name}] "
                f"COMPLETED "
                f"| confidence="
                f"{result['confidence']:.3f}"
            )

            return result

        except Exception as e:

            self._log(
                f"[{agent_name}] ERROR: {e}"
            )

            traceback.print_exc()

            elapsed = (
                datetime.now()
                -
                start_time
            ).total_seconds() * 1000.0

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

                "latency_ms":
                    elapsed,

                "missing_data_ratio":
                    1.0,

                "details":
                    {},

                "explanation":
                    None,

                "error":
                    str(e),
            }

    # =========================================================================
    # SPECIALIZED AGENTS
    # =========================================================================

    def run_specialized_agents(
        self,
        clinical_data=None,
        image=None,
        volume=None,
    ):

        results = {}

        # ---------------------------------------------------------------------
        # CIRRHOSIS
        # ---------------------------------------------------------------------

        if clinical_data is not None:

            results[
                "cirrhosis"
            ] = self._execute_agent(

                "CirrhosisAgent",

                self.cirrhosis_agent,

                clinical_data,

                "cirrhosis_classification",
            )

        # ---------------------------------------------------------------------
        # FATTY LIVER
        # ---------------------------------------------------------------------

        if clinical_data is not None:

            results[
                "fatty_liver"
            ] = self._execute_agent(

                "FattyLiverAgent",

                self.fatty_liver_agent,

                clinical_data,

                "fatty_liver_classification",
            )

        # ---------------------------------------------------------------------
        # FIBROSIS
        # ---------------------------------------------------------------------

        if clinical_data is not None:

            results[
                "fibrosis"
            ] = self._execute_agent(

                "FibrosisAgent",

                self.fibrosis_agent,

                clinical_data,

                "fibrosis_prediction",
            )

        # ---------------------------------------------------------------------
        # TUMOR
        # ---------------------------------------------------------------------

        if image is not None:

            results[
                "tumor_classification"
            ] = self._execute_agent(

                "TumorClassificationAgent",

                self.tumor_agent,

                image,

                "tumor_classification",
            )

        # ---------------------------------------------------------------------
        # SEGMENTATION
        # ---------------------------------------------------------------------

        if volume is not None:

            results[
                "liver_segmentation"
            ] = self._execute_agent(

                "LiverSegmentationAgent",

                self.segmentation_agent,

                volume,

                "liver_segmentation",
            )

        return results

    # =========================================================================
    # STANDARDIZE RESULTS
    # =========================================================================

    def _to_agent_results(
        self,
        raw_results
    ):

        agent_results = []

        for result in raw_results.values():

            agent_result = (
                AgentResult.from_dict(
                    result
                )
            )

            # -------------------------------------------------------------
            # ADAPTIVE TRUST
            # -------------------------------------------------------------

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

            agent_result.trust = trust

            agent_results.append(
                agent_result
            )

        return agent_results

    # =========================================================================
    # CLINICAL REASONING
    # =========================================================================

    def run_clinical_reasoning(
        self,
        agent_results,
        fusion_result=None,
        conflicts=None,
    ):

        if self.clinical_agent is None:

            return {

                "agent_id":
                    "ClinicalReasoningAgent",

                "agent":
                    "ClinicalReasoningAgent",

                "task_type":
                    "clinical_reasoning",

                "status":
                    "not_available",

                "prediction":
                    None,

                "confidence":
                    0.0,

                "uncertainty":
                    1.0,

                "quality":
                    0.0,

                "error":
                    "Clinical reasoning agent not available",
            }

        self._log(
            "\n[ClinicalReasoningAgent] START"
        )

        try:

            # -------------------------------------------------------------
            # Convert AgentResult objects into dictionaries
            # -------------------------------------------------------------

            agent_data = {}

            for result in agent_results:

                data = result.to_dict()

                agent_data[
                    result.agent_id
                ] = data

            # -------------------------------------------------------------
            # Add coordination information
            # -------------------------------------------------------------

            coordination_context = {

                "agents":
                    agent_data,

                "adaptive_fusion":
                    fusion_result or {},

                "conflicts":
                    conflicts or [],
            }

            result = (
                self.clinical_agent.predict(
                    coordination_context
                )
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
                "success"
            )

            result.setdefault(
                "confidence",
                result.get(
                    "probability",
                    0.0
                )
                if isinstance(
                    result.get(
                        "probability"
                    ),
                    (int, float)
                )
                else 0.0
            )

            result.setdefault(
                "uncertainty",
                1.0
                -
                float(
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
                "error",
                None
            )

            self._log(
                "[ClinicalReasoningAgent] "
                "COMPLETED"
            )

            return result

        except Exception as e:

            self._log(
                "[ClinicalReasoningAgent] "
                f"ERROR: {e}"
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

                "confidence":
                    0.0,

                "uncertainty":
                    1.0,

                "quality":
                    0.0,

                "error":
                    str(e),
            }

    # =========================================================================
    # COMPLETE PIPELINE
    # =========================================================================

    def run(
        self,
        patient_id,
        clinical_data=None,
        image=None,
        volume=None,
    ):

        print("\n")
        print("=" * 80)
        print(
            f"LIVERAI ADAPTIVE PIPELINE "
            f"— PATIENT {patient_id}"
        )
        print("=" * 80)

        # ---------------------------------------------------------------------
        # STEP 1
        # SPECIALIZED AGENTS
        # ---------------------------------------------------------------------

        raw_results = (
            self.run_specialized_agents(

                clinical_data=
                    clinical_data,

                image=
                    image,

                volume=
                    volume,
            )
        )

        # ---------------------------------------------------------------------
        # STEP 2
        # STANDARDIZATION + TRUST
        # ---------------------------------------------------------------------

        agent_results = (
            self._to_agent_results(
                raw_results
            )
        )

        # ---------------------------------------------------------------------
        # STEP 3
        # ADAPTIVE EVIDENCE FUSION
        # ---------------------------------------------------------------------

        fusion_result = (
            self.adaptive_fusion.fuse(
                agent_results
            )
        )

        # ---------------------------------------------------------------------
        # STEP 4
        # CONFLICT DETECTION
        # ---------------------------------------------------------------------

        conflicts = (
            self.conflict_detector.detect(
                agent_results
            )
        )

        # ---------------------------------------------------------------------
        # STEP 5
        # CLINICAL REASONING
        # ---------------------------------------------------------------------

        clinical_result = (
            self.run_clinical_reasoning(

                agent_results,

                fusion_result=

                    fusion_result,

                conflicts=
                    conflicts,
            )
        )

        # ---------------------------------------------------------------------
        # STEP 6
        # DECISION INTELLIGENCE
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # STEP 7
        # SERIALIZE AGENTS
        # ---------------------------------------------------------------------

        serialized_agents = {}

        for result in agent_results:

            serialized_agents[
                result.agent_id
            ] = result.to_dict()

        # Add clinical reasoning
        serialized_agents[
            "ClinicalReasoningAgent"
        ] = clinical_result

        # ---------------------------------------------------------------------
        # FINAL RESULT
        # ---------------------------------------------------------------------

        final_result = {

            "patient_id":
                patient_id,

            "status":
                "completed",

            "timestamp":
                datetime.now().isoformat(),

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

        self.last_results = (
            final_result
        )

        print("\n")
        print("=" * 80)
        print(
            "LIVERAI ADAPTIVE PIPELINE "
            "COMPLETED"
        )
        print("=" * 80)

        return final_result

    # =========================================================================
    # UTILITY
    # =========================================================================

    @staticmethod
    def _clip(value):

        try:
            value = float(value)

        except (
            TypeError,
            ValueError
        ):
            value = 0.0

        return max(
            0.0,
            min(
                1.0,
                value
            )
        )
