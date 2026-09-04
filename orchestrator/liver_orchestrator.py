# =============================================================================
# LiverAI-MultiAgent
# COMPLETE MULTI-AGENT ORCHESTRATOR
# =============================================================================

import traceback
from datetime import datetime

from orchestrator.schemas import AgentResult

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
    Main LiverAI coordination layer.

    Agents:

        1. Fatty Liver
        2. Fibrosis
        3. Cirrhosis
        4. Tumor Classification
        5. Liver Segmentation
        6. Clinical Reasoning
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
            "LiverAI Multi-Agent Orchestrator"
        )

        # =========================================================================
        # AGENTS
        # =========================================================================

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

        # =========================================================================
        # REGISTRY
        # =========================================================================

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

        # =========================================================================
        # COORDINATION
        # =========================================================================

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

        # =========================================================================
        # STATE
        # =========================================================================

        self.last_results = {}
        self.last_assessment = None
        self.execution_log = []

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

        print("=" * 80)

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
                "timestamp":
                    timestamp,

                "message":
                    message,
            }
        )

        print(
            message
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

        start_time = datetime.now()

        self._log(
            f"\n[{agent_name}] START"
        )

        # -------------------------------------------------------------------------
        # AGENT NOT AVAILABLE
        # -------------------------------------------------------------------------

        if agent is None:

            self._log(
                f"[{agent_name}] NOT AVAILABLE"
            )

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
                    "Agent not available.",
            }

        # -------------------------------------------------------------------------
        # NO INPUT
        # -------------------------------------------------------------------------

        if input_data is None:

            self._log(
                f"[{agent_name}] NO INPUT → SKIPPED"
            )

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
                    "Required input not provided.",
            }

        # -------------------------------------------------------------------------
        # EXECUTION
        # -------------------------------------------------------------------------

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
                "analyze"
            ):

                result = agent.analyze(
                    input_data
                )

            else:

                raise AttributeError(
                    f"{agent_name} has neither "
                    "`predict()` nor `analyze()`."
                )

            # ---------------------------------------------------------------------
            # NORMALIZE
            # ---------------------------------------------------------------------

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

            # ---------------------------------------------------------------------
            # CLIP VALUES
            # ---------------------------------------------------------------------

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

            result[
                "missing_data_ratio"
            ] = self._clip(
                result.get(
                    "missing_data_ratio",
                    0.0
                )
            )

            result["latency_ms"] = float(
                max(
                    0.0,
                    result.get(
                        "latency_ms",
                        elapsed_ms
                    )
                )
            )

            self._log(
                f"[{agent_name}] COMPLETED "
                f"| confidence="
                f"{result['confidence']:.3f}"
            )

            return result

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

    def run_cirrhosis(
        self,
        clinical_data
    ):

        return self._execute_agent(
            "CirrhosisAgent",
            self.cirrhosis_agent,
            clinical_data,
            "cirrhosis_classification",
        )

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

        results[
            "fatty_liver"
        ] = self.run_fatty_liver(
            clinical_data
        )

        # ---------------------------------------------------------------------
        # 2. FIBROSIS
        # ---------------------------------------------------------------------

        results[
            "fibrosis"
        ] = self.run_fibrosis(
            fibrosis_input
        )

        # ---------------------------------------------------------------------
        # 3. CIRRHOSIS
        # ---------------------------------------------------------------------

        results[
            "cirrhosis"
        ] = self.run_cirrhosis(
            clinical_data
        )

        # ---------------------------------------------------------------------
        # 4. TUMOR
        # ---------------------------------------------------------------------

        results[
            "tumor_classification"
        ] = self.run_tumor_classification(
            image
        )

        # ---------------------------------------------------------------------
        # 5. SEGMENTATION
        # ---------------------------------------------------------------------

        results[
            "liver_segmentation"
        ] = self.run_liver_segmentation(
            volume
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

        agent_results = []

        for result in raw_results.values():

            agent_result = (
                AgentResult.from_dict(
                    result
                )
            )

            # -----------------------------------------------------------------
            # TRUST
            # -----------------------------------------------------------------

            if self.trust_manager is not None:

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
        agent_results,
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
                    "Clinical reasoning agent not available.",
            }

        try:

            # -----------------------------------------------------------------
            # CONVERT AGENTS TO DICTIONARIES
            # -----------------------------------------------------------------

            agent_data = {}

            for result in agent_results:

                agent_data[
                    result.agent_id
                ] = result.to_dict()

            # -----------------------------------------------------------------
            # COORDINATION CONTEXT
            # -----------------------------------------------------------------

            context = {

                "agents":
                    agent_data,

                "adaptive_fusion":
                    fusion_result or {},

                "conflicts":
                    conflicts or [],
            }

            # -----------------------------------------------------------------
            # CALL CLINICAL AGENT
            # -----------------------------------------------------------------

            if hasattr(
                self.clinical_agent,
                "analyze"
            ):

                result = (
                    self.clinical_agent.analyze(
                        context
                    )
                )

            elif hasattr(
                self.clinical_agent,
                "predict"
            ):

                result = (
                    self.clinical_agent.predict(
                        context
                    )
                )

            else:

                raise AttributeError(
                    "Clinical agent must provide "
                    "`analyze()` or `predict()`."
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
                "completed"
            )

            result.setdefault(
                "confidence",
                0.0
            )

            result.setdefault(
                "uncertainty",
                1.0 -
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
                        result.probability,

                    "confidence":
                        result.confidence,

                    "uncertainty":
                        result.uncertainty,

                    "quality":
                        result.quality,

                    "trust":
                        result.trust,
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
                        item["trust"]
                        if item["trust"] is not None
                        else 0.0
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

        # Only compare agents with the SAME task.
        task_groups = {}

        for result in agent_results:

            task_groups.setdefault(
                result.task_type,
                []
            ).append(
                result
            )

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
            }

        confidences = [
            result.confidence
            for result in valid
        ]

        average_confidence = (
            sum(confidences)
            /
            len(confidences)
        )

        # ---------------------------------------------------------------------
        # Simple coordination score.
        #
        # This is NOT a medical diagnosis.
        # ---------------------------------------------------------------------

        risk_components = []

        for result in valid:

            prediction = str(
                result.prediction
            ).lower()

            if any(
                word in prediction
                for word in [
                    "tumor",
                    "carcinoma",
                    "angiosarcoma",
                    "cholangiocarcinoma",
                    "fibrosis",
                    "cirrhosis",
                    "positive",
                    "abnormal",
                ]
            ):

                risk_components.append(
                    result.confidence
                )

        risk_score = (
            sum(risk_components)
            /
            len(risk_components)
            if risk_components
            else 0.0
        )

        if risk_score >= 0.70:

            risk_level = "high"

        elif risk_score >= 0.40:

            risk_level = "moderate"

        else:

            risk_level = "low"

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

            "request_additional_tests":
                bool(
                    len(conflicts) > 0
                ),

            "conflicts_detected":
                len(conflicts),

            "note":
                (
                    "This is a system-level risk "
                    "aggregation and not a medical diagnosis."
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
        # STEP 1 — SPECIALIZED AGENTS
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

                image=
                    image,

                volume=
                    volume,
            )
        )

        # =========================================================================
        # STEP 2 — STANDARDIZATION
        # =========================================================================

        self._log(
            "\nSTEP 2/6 → Standardization + Trust"
        )

        agent_results = (
            self._to_agent_results(
                raw_results
            )
        )

        # =========================================================================
        # STEP 3 — FUSION
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
        # STEP 4 — CONFLICT
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

        clinical_result = (
            self.run_clinical_reasoning(
                agent_results,
                fusion_result=
                    fusion_result,
                conflicts=
                    conflicts,
            )
        )

        # =========================================================================
        # STEP 6 — DECISION
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

        for result in agent_results:

            serialized_agents[
                result.agent_id
            ] = result.to_dict()

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

            "total_specialized_agents":
                5,

            "agents_completed":
                len(
                    completed_agents
                ),

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
