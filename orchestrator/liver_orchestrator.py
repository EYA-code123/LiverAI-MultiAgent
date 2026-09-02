# =============================================================================
# LiverAI-MultiAgent
# ADAPTIVE LIVER AI ORCHESTRATOR
# =============================================================================

import traceback
import time
from datetime import datetime

from orchestrator.schemas import AgentResult
from coordinator.trust import TrustManager
from coordinator.conflict import ConflictDetector
from coordinator.adaptive_fusion import AdaptiveFusion
from coordinator.decision import DecisionEngine


class LiverAIOrchestrator:

    def __init__(
        self,
        cirrhosis_agent=None,
        fatty_liver_agent=None,
        clinical_agent=None,
        fibrosis_agent=None,
        tumor_agent=None,
        segmentation_agent=None
    ):

        self.name = (
            "LiverAI Adaptive Coordinator"
        )

        self.cirrhosis_agent = (
            cirrhosis_agent
        )

        self.fatty_liver_agent = (
            fatty_liver_agent
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

        self.clinical_agent = (
            clinical_agent
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
                clinical_agent
        }

        # ---------------------------------------------------------------------
        # COORDINATION COMPONENTS
        # ---------------------------------------------------------------------

        self.trust_manager = (
            TrustManager()
        )

        self.conflict_detector = (
            ConflictDetector()
        )

        self.adaptive_fusion = (
            AdaptiveFusion()
        )

        self.decision_engine = (
            DecisionEngine()
        )

        self.last_results = {}

        self.execution_log = []

        print("=" * 80)
        print(
            "LIVERAI ADAPTIVE MULTI-AGENT SYSTEM"
        )
        print("=" * 80)

        for name, agent in self.agents.items():

            if agent is not None:

                print(
                    f"✓ {name:<25}"
                    f"{agent.__class__.__name__}"
                )

            else:

                print(
                    f"⚠ {name:<25}"
                    "NOT AVAILABLE"
                )

        print("=" * 80)

    # =========================================================================
    # LOGGING
    # =========================================================================

    def _log(
        self,
        message
    ):

        timestamp = (
            datetime.now()
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        self.execution_log.append({

            "timestamp":
                timestamp,

            "message":
                message
        })

        print(message)

    # =========================================================================
    # STANDARDIZE RESULT
    # =========================================================================

    def _standardize_result(
        self,
        agent_name,
        raw_result,
        latency_ms
    ):

        if raw_result is None:

            raw_result = {}

        if not isinstance(
            raw_result,
            dict
        ):

            raw_result = {
                "prediction":
                    raw_result
            }

        agent_id = raw_result.get(
            "agent_id",
            raw_result.get(
                "agent",
                agent_name
            )
        )

        prediction = raw_result.get(
            "prediction"
        )

        probability = raw_result.get(
            "probability"
        )

        confidence = raw_result.get(
            "confidence"
        )

        if confidence is None:

            if probability is not None:

                try:

                    confidence = float(
                        probability
                    )

                except Exception:

                    confidence = 0.0

            else:

                confidence = 0.0

        uncertainty = raw_result.get(
            "uncertainty"
        )

        if uncertainty is None:

            uncertainty = (
                1.0
                -
                float(confidence)
            )

        quality = raw_result.get(
            "quality",
            1.0
        )

        details = raw_result.get(
            "details",
            {}
        )

        if details is None:

            details = {}

        task_type = raw_result.get(
            "task_type",
            details.get(
                "task_type",
                "unknown"
            )
        )

        missing_data_ratio = (
            raw_result.get(
                "missing_data_ratio",
                details.get(
                    "missing_data_ratio",
                    0.0
                )
            )
        )

        error = raw_result.get(
            "error"
        )

        status = raw_result.get(
            "status",
            "success"
        )

        if status in [
            "completed",
            "ok"
        ]:

            status = "success"

        if error is not None:

            status = "error"

        result = AgentResult(

            agent_id=str(
                agent_id
            ),

            prediction=prediction,

            probability=probability,

            confidence=float(
                confidence
                if confidence is not None
                else 0.0
            ),

            uncertainty=float(
                uncertainty
                if uncertainty is not None
                else 1.0
            ),

            quality=float(
                quality
                if quality is not None
                else 0.0
            ),

            status=status,

            task_type=str(
                task_type
            ),

            latency_ms=float(
                latency_ms
            ),

            missing_data_ratio=float(
                missing_data_ratio
                if missing_data_ratio
                is not None
                else 0.0
            ),

            details=details,

            error=error
        )

        return result

    # =========================================================================
    # EXECUTE AGENT
    # =========================================================================

    def _execute_agent(
        self,
        agent_name,
        agent,
        input_data
    ):

        if agent is None:

            return AgentResult(

                agent_id=
                    agent_name,

                status=
                    "error",

                quality=
                    0.0,

                error=
                    "Agent not available"
            )

        if input_data is None:

            return AgentResult(

                agent_id=
                    agent_name,

                status=
                    "error",

                quality=
                    0.0,

                error=
                    "No input provided"
            )

        self._log(
            f"\n[{agent_name}] START"
        )

        start_time = time.perf_counter()

        try:

            raw_result = agent.predict(
                input_data
            )

            latency_ms = (
                time.perf_counter()
                -
                start_time
            ) * 1000.0

            result = (
                self._standardize_result(
                    agent_name,
                    raw_result,
                    latency_ms
                )
            )

            self._log(
                f"[{agent_name}] "
                f"✓ COMPLETED "
                f"({latency_ms:.2f} ms)"
            )

            return result

        except Exception as e:

            latency_ms = (
                time.perf_counter()
                -
                start_time
            ) * 1000.0

            self._log(
                f"[{agent_name}] "
                f"✗ ERROR: {e}"
            )

            traceback.print_exc()

            return AgentResult(

                agent_id=
                    agent_name,

                status=
                    "error",

                quality=
                    0.0,

                latency_ms=
                    latency_ms,

                error=
                    str(e)
            )

    # =========================================================================
    # SPECIALIZED AGENTS
    # =========================================================================

    def run_specialized_agents(
        self,
        clinical_data=None,
        image=None,
        volume=None
    ):

        results = {}

        # ---------------------------------------------------------------------
        # CIRRHOSIS
        # ---------------------------------------------------------------------

        if clinical_data is not None:

            results["cirrhosis"] = (
                self._execute_agent(

                    "CirrhosisAgent",

                    self.cirrhosis_agent,

                    clinical_data
                )
            )

        # ---------------------------------------------------------------------
        # FATTY LIVER
        # ---------------------------------------------------------------------

        if clinical_data is not None:

            results["fatty_liver"] = (
                self._execute_agent(

                    "FattyLiverAgent",

                    self.fatty_liver_agent,

                    clinical_data
                )
            )

        # ---------------------------------------------------------------------
        # FIBROSIS
        # ---------------------------------------------------------------------

        if clinical_data is not None:

            results["fibrosis"] = (
                self._execute_agent(

                    "FibrosisAgent",

                    self.fibrosis_agent,

                    clinical_data
                )
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

                image
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

                volume
            )

        return results

    # =========================================================================
    # UPDATE TRUST
    # =========================================================================

    def update_agent_trust(
        self,
        results
    ):

        result_list = list(
            results.values()
        )

        # ---------------------------------------------------------------------
        # First pass: calculate agreement
        # ---------------------------------------------------------------------

        agreement = (
            self.conflict_detector
            .agreement_score(
                result_list
            )
        )

        # ---------------------------------------------------------------------
        # Patient-specific trust
        # ---------------------------------------------------------------------

        for result in result_list:

            if not result.success:

                result.trust = 0.0

                continue

            result.trust = (
                self.trust_manager
                .compute_trust(

                    agent_id=
                        result.agent_id,

                    confidence=
                        result.confidence,

                    quality=
                        result.quality,

                    uncertainty=
                        result.uncertainty,

                    missing_data_ratio=
                        result.missing_data_ratio,

                    agreement=
                        agreement
                )
            )

        return results

    # =========================================================================
    # CONFLICT DETECTION
    # =========================================================================

    def detect_conflicts(
        self,
        results
    ):

        return (
            self.conflict_detector
            .detect(
                list(results.values())
            )
        )

    # =========================================================================
    # ADAPTIVE FUSION
    # =========================================================================

    def perform_adaptive_fusion(
        self,
        results
    ):

        return (
            self.adaptive_fusion
            .fuse(
                list(results.values())
            )
        )

    # =========================================================================
    # CLINICAL REASONING
    # =========================================================================

    def run_clinical_reasoning(
        self,
        results
    ):

        if self.clinical_agent is None:

            return {

                "agent":
                    "ClinicalReasoningAgent",

                "status":
                    "not_available",

                "prediction":
                    None,

                "confidence":
                    0.0,

                "risk_score":
                    0.0,

                "error":
                    "Clinical reasoning "
                    "agent not available"
            }

        self._log(
            "\n[ClinicalReasoningAgent] START"
        )

        # ---------------------------------------------------------------------
        # Convert results to dictionaries
        # ---------------------------------------------------------------------

        result_dict = {}

        for key, result in results.items():

            result_dict[key] = (
                result.to_dict()
            )

        try:

            start_time = (
                time.perf_counter()
            )

            raw_result = (
                self.clinical_agent.predict(
                    result_dict
                )
            )

            latency_ms = (
                time.perf_counter()
                -
                start_time
            ) * 1000.0

            if raw_result is None:

                raw_result = {}

            if not isinstance(
                raw_result,
                dict
            ):

                raw_result = {
                    "prediction":
                        raw_result
                }

            raw_result.setdefault(
                "agent",
                "ClinicalReasoningAgent"
            )

            raw_result.setdefault(
                "status",
                "completed"
            )

            raw_result.setdefault(
                "confidence",
                0.0
            )

            raw_result.setdefault(
                "uncertainty",
                1.0
                -
                raw_result.get(
                    "confidence",
                    0.0
                )
            )

            raw_result.setdefault(
                "quality",
                1.0
            )

            raw_result.setdefault(
                "details",
                {}
            )

            raw_result[
                "latency_ms"
            ] = latency_ms

            self._log(
                "[ClinicalReasoningAgent] "
                "✓ COMPLETED"
            )

            return raw_result

        except Exception as e:

            self._log(
                "[ClinicalReasoningAgent] "
                f"✗ ERROR: {e}"
            )

            traceback.print_exc()

            return {

                "agent":
                    "ClinicalReasoningAgent",

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

                "risk_score":
                    0.0,

                "error":
                    str(e)
            }

    # =========================================================================
    # COMPLETE PIPELINE
    # =========================================================================

    def run(
        self,
        patient_id,
        clinical_data=None,
        image=None,
        volume=None
    ):

        print("\n")
        print("=" * 80)
        print(
            "LIVERAI ADAPTIVE PIPELINE"
        )
        print(
            f"PATIENT: {patient_id}"
        )
        print("=" * 80)

        # ---------------------------------------------------------------------
        # STEP 1
        # Specialized agents
        # ---------------------------------------------------------------------

        specialized_results = (
            self.run_specialized_agents(

                clinical_data=
                    clinical_data,

                image=
                    image,

                volume=
                    volume
            )
        )

        # ---------------------------------------------------------------------
        # STEP 2
        # Patient-specific trust
        # ---------------------------------------------------------------------

        specialized_results = (
            self.update_agent_trust(
                specialized_results
            )
        )

        # ---------------------------------------------------------------------
        # STEP 3
        # Conflict detection
        # ---------------------------------------------------------------------

        conflicts = (
            self.detect_conflicts(
                specialized_results
            )
        )

        # ---------------------------------------------------------------------
        # STEP 4
        # Adaptive evidence fusion
        # ---------------------------------------------------------------------

        fusion_result = (
            self.perform_adaptive_fusion(
                specialized_results
            )
        )

        # ---------------------------------------------------------------------
        # STEP 5
        # Clinical reasoning
        # ---------------------------------------------------------------------

        clinical_result = (
            self.run_clinical_reasoning(
                specialized_results
            )
        )

        # ---------------------------------------------------------------------
        # STEP 6
        # Decision intelligence
        # ---------------------------------------------------------------------

        decision = (
            self.decision_engine.decide(

                agent_results=list(
                    specialized_results.values()
                ),

                conflicts=conflicts,

                fusion_result=fusion_result,

                clinical_result=clinical_result
            )
        )

        # ---------------------------------------------------------------------
        # STEP 7
        # FINAL OUTPUT
        # ---------------------------------------------------------------------

        final_agents = {}

        for key, result in (
            specialized_results.items()
        ):

            final_agents[key] = (
                result.to_dict()
            )

        final_agents[
            "clinical_reasoning"
        ] = clinical_result

        final_result = {

            "patient_id":
                patient_id,

            "status":
                "completed",

            "agents":
                final_agents,

            "coordination": {

                "trust": {

                    agent_id:
                        result.trust

                    for agent_id, result
                    in specialized_results.items()
                },

                "conflicts":
                    conflicts,

                "adaptive_fusion":
                    fusion_result
            },

            "decision":
                decision,

            "clinical_reasoning":
                clinical_result,

            "execution_log":
                self.execution_log
        }

        self.last_results = (
            final_result
        )

        print("\n")
        print("=" * 80)
        print(
            "LIVERAI PIPELINE COMPLETED"
        )
        print("=" * 80)

        return final_result
