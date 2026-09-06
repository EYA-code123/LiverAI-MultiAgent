from pathlib import Path

COORDINATOR_PATH = Path(
    "/content/LiverAI-MultiAgent/coordinator/liver_coordinator.py"
)

COORDINATOR_CODE = r'''
# =============================================================================
# LiverAI-MultiAgent
# LIVER AI COORDINATOR
# TASK-AWARE ADAPTIVE COORDINATION
# =============================================================================

from collections import defaultdict
from datetime import datetime


from coordinator.trust_manager import TrustManager
from coordinator.conflict_detector import ConflictDetector
from coordinator.adaptive_fusion import AdaptiveFusion
from coordinator.conflict_resolver import ConflictResolver
from coordinator.reasoning import EvidenceReasoner
from coordinator.decision import DecisionEngine
from coordinator.action import ActionEngine
from coordinator.feedback import FeedbackEngine


class LiverCoordinator:

    # =========================================================================
    # INIT
    # =========================================================================

    def __init__(
        self,
        agents=None,
        minimum_confidence=0.55,
        minimum_coverage=0.50,
        minimum_quality=0.50,
    ):

        self.name = "LiverAICoordinator"

        # ---------------------------------------------------------------------
        # COMPONENTS
        # ---------------------------------------------------------------------

        self.trust_manager = TrustManager()

        self.conflict_detector = ConflictDetector()

        self.fusion = AdaptiveFusion(
            min_confidence=0.0,
            min_quality=0.0,
            use_trust=True
        )

        self.conflict_resolver = ConflictResolver()

        self.reasoner = EvidenceReasoner(
            minimum_confidence=minimum_confidence
        )

        self.decision_engine = DecisionEngine(
            moderate_confidence=minimum_confidence,
            minimum_coverage=minimum_coverage,
            minimum_quality=minimum_quality
        )

        self.action_engine = ActionEngine()

        self.feedback_engine = FeedbackEngine(
            self.trust_manager
        )

        # ---------------------------------------------------------------------
        # AGENTS
        # ---------------------------------------------------------------------

        self.agents = {}

        if agents:
            self._register_initial_agents(agents)

    # =========================================================================
    # INITIAL AGENTS
    # =========================================================================

    def _register_initial_agents(self, agents):

        if isinstance(agents, dict):

            for agent_id, config in agents.items():

                if isinstance(config, dict):

                    agent = config.get("agent")
                    task_type = config.get(
                        "task_type",
                        agent_id
                    )
                    modality = config.get(
                        "modality",
                        "unknown"
                    )

                else:

                    agent = config
                    task_type = agent_id
                    modality = "unknown"

                if agent is not None:

                    self.register_agent(
                        agent_id=agent_id,
                        agent=agent,
                        task_type=task_type,
                        modality=modality
                    )

    # =========================================================================
    # REGISTER
    # =========================================================================

    def register_agent(
        self,
        agent_id,
        agent,
        task_type,
        modality="unknown"
    ):

        if not agent_id:
            raise ValueError(
                "agent_id cannot be empty."
            )

        if agent is None:
            raise ValueError(
                f"Agent '{agent_id}' cannot be None."
            )

        metadata = {

            "agent_id":
                str(agent_id),

            "agent":
                agent,

            "task_type":
                str(task_type),

            "modality":
                str(modality),

            "registered_at":
                datetime.utcnow().isoformat(),

            "status":
                "registered"
        }

        self.agents[str(agent_id)] = metadata

        # Register initial historical trust
        self.trust_manager.register_agent(
            str(agent_id),
            performance=0.5
        )

        return {
            "status": "registered",
            "agent_id": str(agent_id),
            "task_type": str(task_type),
            "modality": str(modality)
        }

    # =========================================================================
    # UNREGISTER
    # =========================================================================

    def unregister_agent(
        self,
        agent_id
    ):

        agent_id = str(agent_id)

        if agent_id not in self.agents:

            return {
                "status": "not_found",
                "agent_id": agent_id
            }

        del self.agents[agent_id]

        return {
            "status": "unregistered",
            "agent_id": agent_id
        }

    # =========================================================================
    # LIST AGENTS
    # =========================================================================

    def list_agents(self):

        output = []

        for agent_id, metadata in self.agents.items():

            output.append({

                "agent_id":
                    agent_id,

                "task_type":
                    metadata["task_type"],

                "modality":
                    metadata["modality"],

                "status":
                    metadata["status"]
            })

        return output

    # =========================================================================
    # GET AGENT
    # =========================================================================

    def get_agent(
        self,
        agent_id
    ):

        metadata = self.agents.get(
            str(agent_id)
        )

        if metadata is None:
            return None

        return metadata["agent"]

    # =========================================================================
    # NORMALIZE RESULT
    # =========================================================================

    def _normalize_result(
        self,
        agent_id,
        result
    ):

        metadata = self.agents[
            str(agent_id)
        ]

        if result is None:

            result = {}

        elif hasattr(result, "to_dict"):

            try:
                result = result.to_dict()
            except Exception:
                result = {}

        elif not isinstance(result, dict):

            result = {
                "prediction": result
            }

        result = dict(result)

        result.setdefault(
            "agent_id",
            str(agent_id)
        )

        result.setdefault(
            "agent",
            str(agent_id)
        )

        result.setdefault(
            "task_type",
            metadata["task_type"]
        )

        result.setdefault(
            "modality",
            metadata["modality"]
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
            0.5
        )

        result.setdefault(
            "missing_data_ratio",
            0.0
        )

        result.setdefault(
            "stability",
            0.5
        )

        result.setdefault(
            "utility",
            0.5
        )

        result.setdefault(
            "probability",
            result.get(
                "confidence"
            )
        )

        result.setdefault(
            "error",
            None
        )

        return result

    # =========================================================================
    # COMPUTE TRUST
    # =========================================================================

    def _compute_trust(
        self,
        result
    ):

        agent_id = result.get(
            "agent_id",
            result.get(
                "agent",
                "unknown"
            )
        )

        confidence = float(
            result.get(
                "confidence",
                0.0
            )
        )

        uncertainty = float(
            result.get(
                "uncertainty",
                1.0 - confidence
            )
        )

        quality = float(
            result.get(
                "quality",
                0.5
            )
        )

        missing_data_ratio = float(
            result.get(
                "missing_data_ratio",
                0.0
            )
        )

        stability = float(
            result.get(
                "stability",
                0.5
            )
        )

        utility = float(
            result.get(
                "utility",
                0.5
            )
        )

        # For a first version, agreement is computed
        # later at task level. Neutral value here.
        agreement = float(
            result.get(
                "agreement",
                0.5
            )
        )

        modality_available = (
            result.get(
                "status"
            )
            not in (
                "not_available",
                "not_run"
            )
        )

        trust = self.trust_manager.compute_trust(

            agent_id=agent_id,

            confidence=confidence,

            uncertainty=uncertainty,

            quality=quality,

            missing_data_ratio=missing_data_ratio,

            agreement=agreement,

            stability=stability,

            utility=utility,

            modality_available=modality_available
        )

        return float(trust)

    # =========================================================================
    # EXECUTE AGENT
    # =========================================================================

    def _execute_agent(
        self,
        agent_id,
        inputs
    ):

        metadata = self.agents[
            str(agent_id)
        ]

        agent = metadata[
            "agent"
        ]

        try:

            # ---------------------------------------------------------------
            # INPUT SELECTION
            # ---------------------------------------------------------------

            if isinstance(inputs, dict):

                if agent_id in inputs:

                    agent_input = inputs[
                        agent_id
                    ]

                elif metadata["task_type"] in inputs:

                    agent_input = inputs[
                        metadata["task_type"]
                    ]

                else:

                    agent_input = inputs

            else:

                agent_input = inputs

            # ---------------------------------------------------------------
            # EXECUTION
            # ---------------------------------------------------------------

            if hasattr(
                agent,
                "predict"
            ):

                result = agent.predict(
                    agent_input
                )

            elif hasattr(
                agent,
                "analyze"
            ):

                result = agent.analyze(
                    agent_input
                )

            elif callable(agent):

                result = agent(
                    agent_input
                )

            else:

                raise TypeError(
                    f"Agent '{agent_id}' has no "
                    "predict(), analyze(), or callable interface."
                )

            return self._normalize_result(
                agent_id,
                result
            )

        except Exception as e:

            return self._normalize_result(
                agent_id,
                {
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
                        str(e)
                }
            )

    # =========================================================================
    # GROUP BY TASK
    # =========================================================================

    def _group_by_task(
        self,
        results
    ):

        groups = defaultdict(list)

        for result in results:

            task = result.get(
                "task_type",
                "unknown"
            )

            groups[
                task
            ].append(
                result
            )

        return dict(groups)

    # =========================================================================
    # GROUP BY MODALITY
    # =========================================================================

    def _group_by_modality(
        self,
        results
    ):

        groups = defaultdict(list)

        for result in results:

            modality = result.get(
                "modality",
                "unknown"
            )

            groups[
                modality
            ].append(
                result
            )

        return dict(groups)

    # =========================================================================
    # TASK AGREEMENT
    # =========================================================================

    def _compute_task_agreement(
        self,
        results
    ):

        valid = [

            r for r in results

            if r.get(
                "prediction"
            ) is not None

            and r.get(
                "status"
            ) in (
                "success",
                "completed"
            )
        ]

        if len(valid) < 2:
            return 1.0

        predictions = [
            str(
                r.get(
                    "prediction"
                )
            )
            for r in valid
        ]

        counts = {}

        for prediction in predictions:

            counts[prediction] = (
                counts.get(
                    prediction,
                    0
                )
                + 1
            )

        majority = max(
            counts.values()
        )

        return float(
            majority /
            len(predictions)
        )

    # =========================================================================
    # UPDATE TASK TRUST
    # =========================================================================

    def _update_task_trust(
        self,
        results
    ):

        groups = self._group_by_task(
            results
        )

        for task_type, task_results in groups.items():

            agreement = (
                self._compute_task_agreement(
                    task_results
                )
            )

            for result in task_results:

                result["agreement"] = (
                    agreement
                )

                try:

                    result["trust"] = (
                        self._compute_trust(
                            result
                        )
                    )

                except Exception:

                    result["trust"] = (
                        0.5
                    )

    # =========================================================================
    # RESOLVE CONFLICTS
    # =========================================================================

    def _resolve_conflicts(
        self,
        results
    ):

        # ConflictDetector is already task-aware.
        conflicts = (
            self.conflict_detector.detect(
                results
            )
        )

        groups = self._group_by_task(
            results
        )

        resolutions = {}

        for task_type, task_results in groups.items():

            task_conflicts = [

                conflict

                for conflict in conflicts

                if conflict.get(
                    "task_type"
                ) == task_type
            ]

            valid = [

                r

                for r in task_results

                if r.get(
                    "prediction"
                ) is not None
            ]

            if len(valid) == 0:
                continue

            if len(valid) == 1:

                resolutions[task_type] = {

                    "status":
                        "resolved",

                    "consensus":
                        True,

                    "prediction":
                        valid[0].get(
                            "prediction"
                        ),

                    "consensus_strength":
                        1.0,

                    "reason":
                        "Single valid agent."
                }

                continue

            try:

                resolutions[
                    task_type
                ] = self.conflict_resolver.resolve(

                    task_type=task_type,

                    results=task_results,

                    conflicts=task_conflicts
                )

            except Exception as e:

                resolutions[
                    task_type
                ] = {

                    "status":
                        "error",

                    "consensus":
                        False,

                    "prediction":
                        None,

                    "consensus_strength":
                        0.0,

                    "reason":
                        str(e)
                }

        return conflicts, resolutions

    # =========================================================================
    # BUILD TASK ASSESSMENTS
    # =========================================================================

    def _build_task_assessments(
        self,
        results,
        resolutions
    ):

        groups = self._group_by_task(
            results
        )

        assessments = {}

        for task_type, task_results in groups.items():

            valid = [

                r

                for r in task_results

                if r.get(
                    "prediction"
                ) is not None

                and r.get(
                    "status"
                ) in (
                    "success",
                    "completed"
                )
            ]

            resolution = resolutions.get(
                task_type
            )

            # ---------------------------------------------------------------
            # Task-local reasoning
            # ---------------------------------------------------------------

            reasoning = self.reasoner.synthesize(

                valid,

                conflict_resolution=resolution
            )

            # ---------------------------------------------------------------
            # Task-local decision
            # ---------------------------------------------------------------

            task_conflicts = [

                c

                for c in (
                    self.conflict_detector.detect(
                        task_results
                    )
                )

                if c.get(
                    "task_type"
                ) == task_type
            ]

            decision = self.decision_engine.decide(

                results=valid,

                conflicts=task_conflicts,

                reasoning=reasoning
            )

            # ---------------------------------------------------------------
            # Normalize naming
            # ---------------------------------------------------------------

            decision_level = decision.get(
                "decision",
                "UNCERTAIN"
            )

            # ---------------------------------------------------------------
            # ActionEngine expects decision_level
            # ---------------------------------------------------------------

            action_input = dict(
                decision
            )

            action_input[
                "decision_level"
            ] = decision_level

            action = self.action_engine.generate(
                action_input
            )

            assessments[
                task_type
            ] = {

                "task_type":
                    task_type,

                "num_agents":
                    len(task_results),

                "num_valid_agents":
                    len(valid),

                "coverage":
                    (
                        len(valid)
                        /
                        len(task_results)
                        if task_results
                        else 0.0
                    ),

                "agreement":
                    self._compute_task_agreement(
                        task_results
                    ),

                "resolution":
                    resolution,

                "reasoning":
                    reasoning,

                "decision":
                    decision,

                "action":
                    action
            }

        return assessments

    # =========================================================================
    # RUN REASONING
    # =========================================================================

    def _run_reasoning(
        self,
        results,
        resolutions
    ):

        assessments = (
            self._build_task_assessments(
                results,
                resolutions
            )
        )

        # Build a global evidence graph only.
        # No global prediction is created.
        evidence_graph = (
            self.reasoner.build_evidence_graph(
                results
            )
        )

        return {

            "status":
                "completed",

            "task_assessments":
                assessments,

            "evidence_graph":
                evidence_graph,

            "prediction":
                None,

            "explanation":
                (
                    "Evidence was synthesized "
                    "independently for each task. "
                    "Predictions from different tasks "
                    "were not merged."
                )
        }

    # =========================================================================
    # RUN DECISION
    # =========================================================================

    def _run_decision(
        self,
        results,
        reasoning
    ):

        assessments = reasoning.get(
            "task_assessments",
            {}
        )

        task_decisions = {}

        for task_type, assessment in assessments.items():

            task_decisions[
                task_type
            ] = assessment.get(
                "decision",
                {}
            )

        # ---------------------------------------------------------------------
        # GLOBAL COORDINATION METRICS
        # ---------------------------------------------------------------------

        total = len(results)

        valid = [

            r

            for r in results

            if r.get(
                "prediction"
            ) is not None

            and r.get(
                "status"
            ) in (
                "success",
                "completed"
            )
        ]

        coverage = (
            len(valid) / total
            if total > 0
            else 0.0
        )

        if valid:

            mean_confidence = sum(

                float(
                    r.get(
                        "confidence",
                        0.0
                    )
                )

                for r in valid

            ) / len(valid)

            mean_trust = sum(

                float(
                    r.get(
                        "trust",
                        0.0
                    )
                )

                for r in valid

            ) / len(valid)

            mean_quality = sum(

                float(
                    r.get(
                        "quality",
                        0.5
                    )
                )

                for r in valid

            ) / len(valid)

        else:

            mean_confidence = 0.0
            mean_trust = 0.0
            mean_quality = 0.0

        # ---------------------------------------------------------------------
        # GLOBAL CONFLICTS ONLY REPRESENT SAME-TASK CONFLICTS
        # ---------------------------------------------------------------------

        conflicts = (
            self.conflict_detector.detect(
                results
            )
        )

        conflict_values = [

            float(
                c.get(
                    "conflict_strength",
                    0.0
                )
            )

            for c in conflicts
        ]

        conflict_score = (

            sum(conflict_values)
            /
            len(conflict_values)

            if conflict_values

            else 0.0
        )

        if not valid:

            coordination_level = "UNCERTAIN"

        elif coverage < 0.50:

            coordination_level = "UNCERTAIN"

        elif conflict_score >= 0.50:

            coordination_level = "UNCERTAIN"

        elif mean_confidence >= 0.80 and mean_trust >= 0.70:

            coordination_level = "HIGH"

        elif mean_confidence >= 0.55:

            coordination_level = "MODERATE"

        else:

            coordination_level = "UNCERTAIN"

        risk_score = (

            0.40 * (
                1.0 -
                mean_confidence
            )

            +

            0.30 * (
                1.0 -
                mean_trust
            )

            +

            0.20 *
            conflict_score

            +

            0.10 * (
                1.0 -
                mean_quality
            )
        )

        risk_score = max(
            0.0,
            min(
                1.0,
                risk_score
            )
        )

        return {

            "status":
                "completed",

            "decision_level":
                coordination_level,

            "task_decisions":
                task_decisions,

            # IMPORTANT:
            # No global prediction.
            "prediction":
                None,

            "confidence":
                float(
                    mean_confidence
                ),

            "uncertainty":
                float(
                    1.0 -
                    mean_confidence
                ),

            "trust":
                float(
                    mean_trust
                ),

            "quality":
                float(
                    mean_quality
                ),

            "coverage":
                float(
                    coverage
                ),

            "conflict_score":
                float(
                    conflict_score
                ),

            "risk_score":
                float(
                    risk_score
                ),

            "request_additional_tests":
                coordination_level ==
                "UNCERTAIN",

            "num_agents":
                total,

            "num_valid_agents":
                len(valid),

            "explanation":
                (
                    "Global coordination metrics "
                    "summarize heterogeneous evidence. "
                    "No cross-task prediction was generated."
                )
        }

    # =========================================================================
    # RUN ACTION
    # =========================================================================

    def _run_action(
        self,
        decision
    ):

        task_actions = {}

        for task_type, task_decision in (
            decision.get(
                "task_decisions",
                {}
            ).items()
        ):

            action_input = dict(
                task_decision
            )

            action_input[
                "decision_level"
            ] = task_decision.get(
                "decision",
                "UNCERTAIN"
            )

            task_actions[
                task_type
            ] = self.action_engine.generate(
                action_input
            )

        return {

            "status":
                "completed",

            "task_actions":
                task_actions,

            "global_action":
                {
                    "status":
                        "coordination_only",

                    "message":
                        (
                            "Clinical actions should "
                            "be interpreted per task."
                        )
                }
        }

    # =========================================================================
    # TASK SUMMARY
    # =========================================================================

    def _build_task_summary(
        self,
        results
    ):

        groups = self._group_by_task(
            results
        )

        summary = {}

        for task_type, task_results in groups.items():

            valid = [

                r

                for r in task_results

                if r.get(
                    "prediction"
                ) is not None

                and r.get(
                    "status"
                ) in (
                    "success",
                    "completed"
                )
            ]

            summary[
                task_type
            ] = {

                "agents":
                    [
                        r.get(
                            "agent_id",
                            r.get(
                                "agent"
                            )
                        )

                        for r in task_results
                    ],

                "valid_agents":
                    [
                        r.get(
                            "agent_id",
                            r.get(
                                "agent"
                            )
                        )

                        for r in valid
                    ],

                "num_agents":
                    len(task_results),

                "num_valid_agents":
                    len(valid),

                "predictions":
                    [
                        r.get(
                            "prediction"
                        )

                        for r in valid
                    ],

                "confidence":
                    [
                        float(
                            r.get(
                                "confidence",
                                0.0
                            )
                        )

                        for r in valid
                    ]
            }

        return summary

    # =========================================================================
    # RUN
    # =========================================================================

    def run(
        self,
        patient_id,
        inputs=None
    ):

        inputs = (
            inputs
            if inputs is not None
            else {}
        )

        started_at = (
            datetime.utcnow()
        )

        results = []

        # ---------------------------------------------------------------------
        # EXECUTE ALL REGISTERED AGENTS
        # ---------------------------------------------------------------------

        for agent_id in self.agents:

            result = self._execute_agent(
                agent_id,
                inputs
            )

            results.append(
                result
            )

        # ---------------------------------------------------------------------
        # TRUST
        # ---------------------------------------------------------------------

        self._update_task_trust(
            results
        )

        # ---------------------------------------------------------------------
        # CONFLICT
        # ---------------------------------------------------------------------

        conflicts, resolutions = (
            self._resolve_conflicts(
                results
            )
        )

        # ---------------------------------------------------------------------
        # FUSION
        # ---------------------------------------------------------------------

        fusion_result = self.fusion.fuse(
            results
        )

        # ---------------------------------------------------------------------
        # REASONING
        # ---------------------------------------------------------------------

        reasoning = self._run_reasoning(
            results,
            resolutions
        )

        # ---------------------------------------------------------------------
        # DECISION
        # ---------------------------------------------------------------------

        decision = self._run_decision(
            results,
            reasoning
        )

        # ---------------------------------------------------------------------
        # ACTION
        # ---------------------------------------------------------------------

        action = self._run_action(
            decision
        )

        # ---------------------------------------------------------------------
        # TASK SUMMARY
        # ---------------------------------------------------------------------

        task_summary = (
            self._build_task_summary(
                results
            )
        )

        elapsed_ms = (
            datetime.utcnow()
            - started_at
        ).total_seconds() * 1000.0

        return {

            "status":
                "completed",

            "patient_id":
                patient_id,

            "coordinator":
                self.name,

            "timestamp":
                started_at.isoformat(),

            "latency_ms":
                float(
                    elapsed_ms
                ),

            "num_registered_agents":
                len(
                    self.agents
                ),

            "agent_results":
                results,

            "task_summary":
                task_summary,

            "conflicts":
                conflicts,

            "conflict_resolutions":
                resolutions,

            "fusion":
                fusion_result,

            "reasoning":
                reasoning,

            "decision":
                decision,

            "action":
                action
        }

    # =========================================================================
    # FEEDBACK
    # =========================================================================

    def feedback(
        self,
        agent_results,
        ground_truth
    ):

        if isinstance(
            agent_results,
            dict
        ):

            results = list(
                agent_results.values()
            )

        else:

            results = list(
                agent_results
            )

        return self.feedback_engine.update(
            results,
            ground_truth
        )

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self):

        agents_status = {}

        for agent_id, metadata in (
            self.agents.items()
        ):

            agent = metadata[
                "agent"
            ]

            try:

                if hasattr(
                    agent,
                    "health_check"
                ):

                    status = agent.health_check()

                else:

                    status = {
                        "status":
                            "available"
                    }

            except Exception as e:

                status = {

                    "status":
                        "error",

                    "error":
                        str(e)
                }

            agents_status[
                agent_id
            ] = {

                "task_type":
                    metadata[
                        "task_type"
                    ],

                "modality":
                    metadata[
                        "modality"
                    ],

                "health":
                    status
            }

        return {

            "coordinator":
                self.name,

            "status":
                "healthy",

            "num_agents":
                len(
                    self.agents
                ),

            "agents":
                agents_status
        }

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

        except Exception:

            return 0.0

        return max(
            0.0,
            min(
                1.0,
                value
            )
        )

    @staticmethod
    def _same_value(
        a,
        b
    ):

        return str(
            a
        ).strip().lower() == str(
            b
        ).strip().lower()


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

LiverAICoordinator = LiverCoordinator
'''

COORDINATOR_PATH.write_text(
    COORDINATOR_CODE,
    encoding="utf-8"
)

print("✅ liver_coordinator.py reconstruit")
print("📁", COORDINATOR_PATH)
print("📦 Taille :", COORDINATOR_PATH.stat().st_size, "bytes")

# ============================================================================
# VÉRIFICATION
# ============================================================================

text = COORDINATOR_PATH.read_text(
    encoding="utf-8"
)

print()
print("===== VÉRIFICATION =====")
print(
    "LiverCoordinator :",
    "class LiverCoordinator" in text
)
print(
    "list_agents :",
    "def list_agents(" in text
)
print(
    "get_agent :",
    "def get_agent(" in text
)
print(
    "register_agent :",
    "def register_agent(" in text
)
print(
    "run :",
    "def run(" in text
)
print(
    "task-aware :",
    "_group_by_task" in text
)
print(
    "alias :",
    "LiverAICoordinator = LiverCoordinator" in text
)
