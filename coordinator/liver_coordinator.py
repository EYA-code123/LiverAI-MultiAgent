# =============================================================================
# LiverAI-MultiAgent
# LIVER AI COORDINATOR
# TASK-AWARE ADAPTIVE COORDINATION
# =============================================================================

from collections import defaultdict
from datetime import datetime


# =============================================================================
# COORDINATION MODULES
# =============================================================================

from coordinator.trust import TrustManager
from coordinator.conflict import ConflictDetector
from coordinator.adaptive_fusion import AdaptiveFusion
from coordinator.conflict_resolver import ConflictResolver
from coordinator.evidence_reasoner import EvidenceReasoner
from coordinator.decision import DecisionEngine
from coordinator.action import ActionEngine
from coordinator.feedback import FeedbackEngine


# =============================================================================
# LIVER COORDINATOR
# =============================================================================

class LiverCoordinator:
    """
    Central task-aware coordination engine for LiverAI-MultiAgent.

    Architecture
    ------------

        Agent Inputs
             |
             v
        Agent Execution
             |
             v
        Result Normalization
             |
             v
        Adaptive Trust
             |
             v
        Task-Aware Fusion
             |
             v
        Same-Task Conflict Detection
             |
             v
        Task-Specific Conflict Resolution
             |
             v
        Evidence Reasoning
             |
             v
        Coordination Decision
             |
             v
        Action Generation


    IMPORTANT
    ---------

    Agents performing different tasks are NOT directly compared.

    Example:

        CirrhosisAgent
            task_type = cirrhosis_classification

        ClinicalReasoningAgent
            task_type = clinical_reasoning

    Their numerical predictions are NOT assumed to represent
    the same target.

    The coordinator therefore preserves task-specific predictions
    and produces a global coordination decision without creating
    an artificial cross-task prediction.
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, agents=None):

        self.name = "LiverAI Adaptive Coordinator"

        # ---------------------------------------------------------------------
        # Registered agents
        # ---------------------------------------------------------------------

        self.agents = {}

        # ---------------------------------------------------------------------
        # Coordination modules
        # ---------------------------------------------------------------------

        self.trust_manager = TrustManager()

        self.conflict_detector = ConflictDetector()

        self.adaptive_fusion = AdaptiveFusion()

        self.conflict_resolver = ConflictResolver()

        self.evidence_reasoner = EvidenceReasoner()

        self.decision_engine = DecisionEngine()

        self.action_engine = ActionEngine()

        self.feedback_engine = FeedbackEngine()

        # ---------------------------------------------------------------------
        # Runtime state
        # ---------------------------------------------------------------------

        self.last_result = None

        self.execution_history = []

        # ---------------------------------------------------------------------
        # Optional initial agents
        # ---------------------------------------------------------------------

        if agents is not None:
            self._register_initial_agents(agents)

    # =========================================================================
    # INITIAL AGENTS
    # =========================================================================

    def _register_initial_agents(self, agents):
        """
        Register agents supplied during initialization.
        """

        if isinstance(agents, dict):

            for agent_id, config in agents.items():

                if isinstance(config, dict):

                    agent = config.get("agent")

                    task_type = config.get(
                        "task_type",
                        "unknown"
                    )

                    modality = config.get(
                        "modality",
                        "unknown"
                    )

                else:

                    agent = config
                    task_type = "unknown"
                    modality = "unknown"

                self.register_agent(
                    agent_id=agent_id,
                    agent=agent,
                    task_type=task_type,
                    modality=modality
                )

        elif isinstance(agents, (list, tuple)):

            for item in agents:

                if not isinstance(item, dict):
                    continue

                agent = item.get("agent")

                agent_id = item.get(
                    "agent_id",
                    getattr(
                        agent,
                        "AGENT_NAME",
                        type(agent).__name__
                        if agent is not None
                        else "unknown"
                    )
                )

                self.register_agent(
                    agent_id=agent_id,
                    agent=agent,
                    task_type=item.get(
                        "task_type",
                        "unknown"
                    ),
                    modality=item.get(
                        "modality",
                        "unknown"
                    )
                )

        else:

            raise TypeError(
                "agents must be a dictionary, list, tuple, or None."
            )

    # =========================================================================
    # REGISTER AGENT
    # =========================================================================

    def register_agent(
        self,
        agent_id,
        agent,
        task_type,
        modality="unknown"
    ):
        """
        Register a specialized agent.
        """

        if agent is None:

            raise ValueError(
                f"Cannot register empty agent: {agent_id}"
            )

        if not hasattr(agent, "predict") and not hasattr(
            agent,
            "analyze"
        ):

            raise TypeError(
                f"Agent '{agent_id}' must expose "
                "predict() or analyze()."
            )

        agent_id = str(agent_id)
        task_type = str(task_type)
        modality = str(modality)

        self.agents[agent_id] = {

            "agent_id": agent_id,

            "agent": agent,

            "task_type": task_type,

            "modality": modality,
        }

        return {

            "status": "registered",

            "agent_id": agent_id,

            "task_type": task_type,

            "modality": modality,
        }

    # =========================================================================
    # UNREGISTER AGENT
    # =========================================================================

    def unregister_agent(self, agent_id):

        agent_id = str(agent_id)

        if agent_id in self.agents:

            del self.agents[agent_id]

            return {

                "status": "unregistered",

                "agent_id": agent_id,
            }

        return {

            "status": "not_found",

            "agent_id": agent_id,
        }

    # =========================================================================
    # LIST AGENTS
    # =========================================================================

    def list_agents(self):
        """
        Return metadata for all registered agents.
        """

        return {

            agent_id: {

                "task_type":
                    config["task_type"],

                "modality":
                    config["modality"],

                "agent":
                    type(
                        config["agent"]
                    ).__name__,

            }

            for agent_id, config in self.agents.items()
        }

    # =========================================================================
    # GET AGENT
    # =========================================================================

    def get_agent(self, agent_id):

        return self.agents.get(
            str(agent_id)
        )

    # =========================================================================
    # NORMALIZE RESULT
    # =========================================================================

    def _normalize_result(
        self,
        result,
        agent_id,
        task_type,
        modality
    ):
        """
        Normalize heterogeneous agent outputs.
        """

        if result is None:
            result = {}

        if not isinstance(result, dict):

            result = {
                "prediction": result
            }

        result = dict(result)

        # ---------------------------------------------------------------------
        # Identity
        # ---------------------------------------------------------------------

        result["agent_id"] = str(
            result.get(
                "agent_id",
                agent_id
            )
        )

        result["agent"] = str(
            result.get(
                "agent",
                agent_id
            )
        )

        # ---------------------------------------------------------------------
        # Task metadata
        # ---------------------------------------------------------------------

        result["task_type"] = str(
            result.get(
                "task_type",
                task_type
            )
        )

        result["modality"] = str(
            result.get(
                "modality",
                modality
            )
        )

        # ---------------------------------------------------------------------
        # Status
        # ---------------------------------------------------------------------

        result["status"] = str(
            result.get(
                "status",
                "success"
            )
        )

        # ---------------------------------------------------------------------
        # Prediction
        # ---------------------------------------------------------------------

        result.setdefault(
            "prediction",
            None
        )

        # ---------------------------------------------------------------------
        # Confidence
        # ---------------------------------------------------------------------

        confidence = result.get(
            "confidence"
        )

        if confidence is None:

            probability = result.get(
                "probability"
            )

            if isinstance(
                probability,
                (int, float)
            ):

                confidence = probability

            else:

                confidence = 0.0

        result["confidence"] = self._clip(
            confidence
        )

        # ---------------------------------------------------------------------
        # Uncertainty
        # ---------------------------------------------------------------------

        if result.get("uncertainty") is None:

            result["uncertainty"] = (
                1.0
                -
                result["confidence"]
            )

        else:

            result["uncertainty"] = self._clip(
                result["uncertainty"]
            )

        # ---------------------------------------------------------------------
        # Quality
        # ---------------------------------------------------------------------

        default_quality = (
            1.0
            if result["status"]
            in (
                "success",
                "completed"
            )
            else 0.0
        )

        result["quality"] = self._clip(
            result.get(
                "quality",
                default_quality
            )
        )

        # ---------------------------------------------------------------------
        # Missing data
        # ---------------------------------------------------------------------

        result["missing_data_ratio"] = self._clip(
            result.get(
                "missing_data_ratio",
                0.0
            )
        )

        # ---------------------------------------------------------------------
        # Stability
        # ---------------------------------------------------------------------

        result["stability"] = self._clip(
            result.get(
                "stability",
                0.5
            )
        )

        # ---------------------------------------------------------------------
        # Utility
        # ---------------------------------------------------------------------

        result["utility"] = self._clip(
            result.get(
                "utility",
                0.5
            )
        )

        # ---------------------------------------------------------------------
        # Error
        # ---------------------------------------------------------------------

        result.setdefault(
            "error",
            None
        )

        if result["error"]:

            result["status"] = "error"

        # ---------------------------------------------------------------------
        # Explanation
        # ---------------------------------------------------------------------

        result.setdefault(
            "explanation",
            None
        )

        # ---------------------------------------------------------------------
        # Latency
        # ---------------------------------------------------------------------

        try:

            result["latency_ms"] = float(
                max(
                    0.0,
                    result.get(
                        "latency_ms",
                        0.0
                    )
                )
            )

        except Exception:

            result["latency_ms"] = 0.0

        return result

    # =========================================================================
    # TRUST
    # =========================================================================

    def _compute_trust(self, result):
        """
        Compute adaptive patient-specific trust.
        """

        try:

            trust = self.trust_manager.compute_trust(

                agent_id=result["agent_id"],

                confidence=result["confidence"],

                quality=result["quality"],

                uncertainty=result["uncertainty"],

                missing_data_ratio=result[
                    "missing_data_ratio"
                ]
            )

        except TypeError:

            try:

                trust = self.trust_manager.compute_trust(

                    result["agent_id"],

                    result["confidence"],

                    result["quality"],

                    result["uncertainty"],

                    result["missing_data_ratio"]
                )

            except Exception:

                trust = 0.0

        except Exception:

            trust = 0.0

        result["trust"] = self._clip(
            trust
        )

        return result

    # =========================================================================
    # AGENT EXECUTION
    # =========================================================================

    def _execute_agent(
        self,
        agent_id,
        config,
        input_data
    ):
        """
        Safely execute one agent.
        """

        task_type = config[
            "task_type"
        ]

        modality = config[
            "modality"
        ]

        agent = config[
            "agent"
        ]

        # ---------------------------------------------------------------------
        # Missing input
        # ---------------------------------------------------------------------

        if input_data is None:

            return self._normalize_result(

                {

                    "status":
                        "no_input",

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

                    "error":
                        "No input provided.",
                },

                agent_id,

                task_type,

                modality
            )

        # ---------------------------------------------------------------------
        # Execute
        # ---------------------------------------------------------------------

        start_time = datetime.now()

        try:

            if hasattr(agent, "predict"):

                raw_result = agent.predict(
                    input_data
                )

            elif hasattr(agent, "analyze"):

                raw_result = agent.analyze(
                    input_data
                )

            else:

                raise TypeError(
                    f"Agent '{agent_id}' has no "
                    "predict() or analyze() method."
                )

            elapsed_ms = (
                datetime.now()
                -
                start_time
            ).total_seconds() * 1000.0

            result = self._normalize_result(

                raw_result,

                agent_id,

                task_type,

                modality
            )

            if result.get(
                "latency_ms",
                0.0
            ) <= 0:

                result["latency_ms"] = elapsed_ms

            return result

        except Exception as e:

            elapsed_ms = (
                datetime.now()
                -
                start_time
            ).total_seconds() * 1000.0

            return self._normalize_result(

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

                    "missing_data_ratio":
                        1.0,

                    "latency_ms":
                        elapsed_ms,

                    "error":
                        f"{type(e).__name__}: {e}",
                },

                agent_id,

                task_type,

                modality
            )

    # =========================================================================
    # GROUP BY TASK
    # =========================================================================

    @staticmethod
    def _group_by_task(results):

        groups = defaultdict(list)

        for result in results:

            if not isinstance(
                result,
                dict
            ):

                continue

            task_type = str(
                result.get(
                    "task_type",
                    "unknown"
                )
            )

            groups[
                task_type
            ].append(result)

        return dict(groups)

    # =========================================================================
    # GROUP BY MODALITY
    # =========================================================================

    @staticmethod
    def _group_by_modality(results):

        groups = defaultdict(list)

        for result in results:

            if not isinstance(
                result,
                dict
            ):

                continue

            modality = str(
                result.get(
                    "modality",
                    "unknown"
                )
            )

            groups[
                modality
            ].append(result)

        return dict(groups)

    # =========================================================================
    # CONFLICT RESOLUTION
    # =========================================================================

    def _resolve_conflicts(
        self,
        results,
        conflicts
    ):
        """
        Resolve conflicts only among agents performing the same task.
        """

        if not conflicts:

            return {}

        task_groups = self._group_by_task(
            results
        )

        resolutions = {}

        conflict_tasks = set()

        for conflict in conflicts:

            if not isinstance(
                conflict,
                dict
            ):

                continue

            conflict_tasks.add(
                str(
                    conflict.get(
                        "task_type",
                        "unknown"
                    )
                )
            )

        for task_type in conflict_tasks:

            task_results = task_groups.get(
                task_type,
                []
            )

            if not task_results:

                continue

            task_conflicts = [

                conflict

                for conflict in conflicts

                if str(
                    conflict.get(
                        "task_type",
                        "unknown"
                    )
                )
                ==
                task_type
            ]

            # -----------------------------------------------------------------
            # Try modern signature
            # -----------------------------------------------------------------

            try:

                resolution = (
                    self.conflict_resolver.resolve(
                        task_results
                    )
                )

            except TypeError:

                try:

                    resolution = (
                        self.conflict_resolver.resolve(

                            conflicts=
                                task_conflicts,

                            results=
                                task_results
                        )
                    )

                except Exception as e:

                    resolution = {

                        "status":
                            "resolution_failed",

                        "task_type":
                            task_type,

                        "error":
                            str(e),
                    }

            except Exception as e:

                resolution = {

                    "status":
                        "resolution_failed",

                    "task_type":
                        task_type,

                    "error":
                        str(e),
                }

            if not isinstance(
                resolution,
                dict
            ):

                resolution = {

                    "status":
                        "resolved",

                    "task_type":
                        task_type,

                    "resolution":
                        resolution,
                }

            resolution.setdefault(
                "task_type",
                task_type
            )

            resolutions[
                task_type
            ] = resolution

        return resolutions

    # =========================================================================
    # TASK ASSESSMENTS
    # =========================================================================

    def _build_task_assessments(
        self,
        results,
        fusion_result,
        conflict_resolutions
    ):
        """
        Build one independent assessment per task.

        No numerical voting occurs across different tasks.
        """

        groups = self._group_by_task(
            results
        )

        task_assessments = {}

        fusion_tasks = {}

        if isinstance(
            fusion_result,
            dict
        ):

            fusion_tasks = fusion_result.get(
                "same_task_fusion",
                {}
            )

            if not isinstance(
                fusion_tasks,
                dict
            ):

                fusion_tasks = {}

        for task_type, items in groups.items():

            valid = [

                item

                for item in items

                if item.get("status")
                in (
                    "success",
                    "completed"
                )

                and item.get(
                    "prediction"
                ) is not None
            ]

            # -----------------------------------------------------------------
            # No valid evidence
            # -----------------------------------------------------------------

            if not valid:

                task_assessments[
                    task_type
                ] = {

                    "task_type":
                        task_type,

                    "status":
                        "no_valid_evidence",

                    "prediction":
                        None,

                    "confidence":
                        0.0,

                    "uncertainty":
                        1.0,

                    "num_agents":
                        len(items),

                    "num_valid_agents":
                        0,

                    "supporting_agents":
                        [],

                    "predictions":
                        [],

                    "conflict_resolution":
                        conflict_resolutions.get(
                            task_type
                        ),
                }

                continue

            # -----------------------------------------------------------------
            # Same-task fusion
            # -----------------------------------------------------------------

            fused = fusion_tasks.get(
                task_type
            )

            if isinstance(
                fused,
                dict
            ):

                prediction = fused.get(
                    "predicted_class_index"
                )

                if prediction is None:

                    prediction = fused.get(
                        "prediction"
                    )

                confidence = self._clip(
                    fused.get(
                        "confidence",
                        0.0
                    )
                )

            else:

                # -------------------------------------------------------------
                # Best trusted agent
                # -------------------------------------------------------------

                best = max(

                    valid,

                    key=lambda x: (

                        float(
                            x.get(
                                "trust",
                                0.0
                            )
                        ),

                        float(
                            x.get(
                                "confidence",
                                0.0
                            )
                        )
                    )
                )

                prediction = best.get(
                    "prediction"
                )

                confidence = self._clip(
                    best.get(
                        "confidence",
                        0.0
                    )
                )

            # -----------------------------------------------------------------
            # Assessment
            # -----------------------------------------------------------------

            task_assessments[
                task_type
            ] = {

                "task_type":
                    task_type,

                "status":
                    "completed",

                "prediction":
                    prediction,

                "confidence":
                    confidence,

                "uncertainty":
                    1.0 - confidence,

                "num_agents":
                    len(items),

                "num_valid_agents":
                    len(valid),

                "supporting_agents": [

                    item["agent_id"]

                    for item in valid
                ],

                "predictions": [

                    item.get(
                        "prediction"
                    )

                    for item in valid
                ],

                "conflict_resolution":
                    conflict_resolutions.get(
                        task_type
                    ),
            }

        return task_assessments

    # =========================================================================
    # TASK-AWARE REASONING
    # =========================================================================

    def _run_reasoning(
        self,
        results,
        fusion_result,
        conflicts,
        conflict_resolutions
    ):
        """
        Generate task-aware evidence reasoning.

        Different tasks are represented as complementary evidence,
        not as direct numerical votes.
        """

        task_assessments = (
            self._build_task_assessments(

                results,

                fusion_result,

                conflict_resolutions
            )
        )

        # ---------------------------------------------------------------------
        # Evidence nodes
        # ---------------------------------------------------------------------

        nodes = []

        for result in results:

            if result.get(
                "status"
            ) not in (
                "success",
                "completed"
            ):

                continue

            node_id = (

                f"{result['agent_id']}:"

                f"{result.get('prediction')}"
            )

            nodes.append({

                "id":
                    node_id,

                "agent":
                    result["agent_id"],

                "task":
                    result["task_type"],

                "modality":
                    result.get(
                        "modality",
                        "unknown"
                    ),

                "prediction":
                    result.get(
                        "prediction"
                    ),

                "confidence":
                    result.get(
                        "confidence",
                        0.0
                    ),

                "trust":
                    result.get(
                        "trust",
                        0.0
                    ),

                "quality":
                    result.get(
                        "quality",
                        0.0
                    ),
            })

        # ---------------------------------------------------------------------
        # Evidence edges
        # ---------------------------------------------------------------------

        edges = []

        for i in range(
            len(nodes)
        ):

            for j in range(
                i + 1,
                len(nodes)
            ):

                a = nodes[i]
                b = nodes[j]

                if a["task"] == b["task"]:

                    if self._same_value(
                        a["prediction"],
                        b["prediction"]
                    ):

                        relation = "supports"

                    else:

                        relation = "conflicts"

                else:

                    relation = "complements"

                edges.append({

                    "source":
                        a["id"],

                    "target":
                        b["id"],

                    "relation":
                        relation,

                    "source_task":
                        a["task"],

                    "target_task":
                        b["task"],
                })

        # ---------------------------------------------------------------------
        # EvidenceReasoner
        # ---------------------------------------------------------------------

        reasoning = None

        try:

            reasoning = (
                self.evidence_reasoner.synthesize(

                    results,

                    fusion_result,

                    conflicts,

                    conflict_resolutions
                )
            )

        except TypeError:

            try:

                reasoning = (
                    self.evidence_reasoner.synthesize(
                        results
                    )
                )

            except Exception:

                reasoning = None

        except Exception:

            reasoning = None

        if not isinstance(
            reasoning,
            dict
        ):

            reasoning = {}

        # ---------------------------------------------------------------------
        # Remove potentially unsafe global prediction
        # ---------------------------------------------------------------------

        reasoning.pop(
            "prediction",
            None
        )

        reasoning.pop(
            "clinical_prediction",
            None
        )

        # ---------------------------------------------------------------------
        # Task-aware metadata
        # ---------------------------------------------------------------------

        reasoning["status"] = "completed"

        reasoning["task_assessments"] = (
            task_assessments
        )

        reasoning["evidence_graph"] = {

            "nodes":
                nodes,

            "edges":
                edges,
        }

        reasoning["num_tasks"] = len(
            task_assessments
        )

        reasoning["num_evidence_sources"] = len(
            nodes
        )

        reasoning["conflicts"] = conflicts

        reasoning["conflict_resolutions"] = (
            conflict_resolutions
        )

        # ---------------------------------------------------------------------
        # Human-readable explanation
        # ---------------------------------------------------------------------

        explanations = []

        for task_type, assessment in (
            task_assessments.items()
        ):

            prediction = assessment.get(
                "prediction"
            )

            confidence = assessment.get(
                "confidence",
                0.0
            )

            agents = assessment.get(
                "supporting_agents",
                []
            )

            explanations.append(

                f"{task_type}: "

                f"prediction={prediction}, "

                f"confidence={confidence:.3f}, "

                f"supported_by={agents}"
            )

        if explanations:

            reasoning["explanation"] = (

                "Task-aware evidence assessment. "

                +

                " | ".join(
                    explanations
                )
            )

        else:

            reasoning["explanation"] = (
                "No valid task-specific evidence available."
            )

        return reasoning

    # =========================================================================
    # DECISION
    # =========================================================================

    def _run_decision(
        self,
        results,
        fusion_result,
        conflicts,
        reasoning
    ):
        """
        Produce a global coordination decision.

        IMPORTANT:
        The returned decision is a coordination level, NOT a prediction
        shared across heterogeneous tasks.
        """

        successful = [

            result

            for result in results

            if result.get("status")
            in (
                "success",
                "completed"
            )

            and result.get(
                "prediction"
            ) is not None
        ]

        total = len(
            results
        )

        valid_count = len(
            successful
        )

        coverage = (

            valid_count / total

            if total > 0

            else 0.0
        )

        # ---------------------------------------------------------------------
        # Weighted confidence and trust
        # ---------------------------------------------------------------------

        if successful:

            weights = [

                max(
                    0.0,
                    float(
                        result.get(
                            "trust",
                            0.0
                        )
                    )
                )

                for result in successful
            ]

            total_weight = sum(
                weights
            )

            if total_weight > 0:

                weighted_confidence = (

                    sum(

                        float(
                            result.get(
                                "confidence",
                                0.0
                            )
                        )
                        *
                        weight

                        for result, weight
                        in zip(
                            successful,
                            weights
                        )
                    )

                    /

                    total_weight
                )

                weighted_trust = (

                    sum(

                        float(
                            result.get(
                                "trust",
                                0.0
                            )
                        )
                        *
                        weight

                        for result, weight
                        in zip(
                            successful,
                            weights
                        )
                    )

                    /

                    total_weight
                )

            else:

                weighted_confidence = 0.0

                weighted_trust = 0.0

        else:

            weighted_confidence = 0.0

            weighted_trust = 0.0

        weighted_confidence = self._clip(
            weighted_confidence
        )

        weighted_trust = self._clip(
            weighted_trust
        )

        # ---------------------------------------------------------------------
        # Conflict score
        #
        # Only conflicts reported by the conflict detector are considered.
        # Since the detector should be task-aware, heterogeneous tasks are
        # not counted as conflicts.
        # ---------------------------------------------------------------------

        if conflicts:

            strengths = []

            for conflict in conflicts:

                try:

                    strength = float(
                        conflict.get(
                            "conflict_strength",
                            0.0
                        )
                    )

                except Exception:

                    strength = 0.0

                strengths.append(
                    self._clip(
                        strength
                    )
                )

            conflict_score = (

                sum(strengths)
                /
                len(strengths)

                if strengths

                else 0.0
            )

        else:

            conflict_score = 0.0

        conflict_score = self._clip(
            conflict_score
        )

        # ---------------------------------------------------------------------
        # Coordination confidence
        #
        # This is intentionally a coordination metric, not a class probability.
        # ---------------------------------------------------------------------

        coordination_confidence = (

            0.35
            *
            weighted_confidence

            +

            0.25
            *
            weighted_trust

            +

            0.20
            *
            coverage

            +

            0.20
            *
            (
                1.0
                -
                conflict_score
            )
        )

        coordination_confidence = self._clip(
            coordination_confidence
        )

        # ---------------------------------------------------------------------
        # Additional evidence
        # ---------------------------------------------------------------------

        request_additional_tests = False

        reasons = []

        if coverage < 0.50:

            request_additional_tests = True

            reasons.append(
                "Insufficient valid agent coverage."
            )

        if weighted_confidence < 0.55:

            request_additional_tests = True

            reasons.append(
                "Overall evidence confidence is low."
            )

        if conflict_score >= 0.50:

            request_additional_tests = True

            reasons.append(
                "Strong conflict detected among compatible "
                "same-task agents."
            )

        # ---------------------------------------------------------------------
        # Decision level
        # ---------------------------------------------------------------------

        if not successful:

            decision_level = (
                "INSUFFICIENT_EVIDENCE"
            )

        elif request_additional_tests:

            decision_level = (
                "UNCERTAIN"
            )

        elif coordination_confidence >= 0.80:

            decision_level = "HIGH"

        elif coordination_confidence >= 0.55:

            decision_level = "MODERATE"

        else:

            decision_level = "UNCERTAIN"

        # ---------------------------------------------------------------------
        # Try DecisionEngine
        # ---------------------------------------------------------------------

        engine_decision = {}

        try:

            engine_decision = (
                self.decision_engine.decide(

                    agent_results=
                        results,

                    conflicts=
                        conflicts,

                    fused_results=
                        fusion_result,

                    clinical_reasoning=
                        reasoning
                )
            )

        except Exception:

            engine_decision = {}

        if not isinstance(
            engine_decision,
            dict
        ):

            engine_decision = {}

        # ---------------------------------------------------------------------
        # Remove predictions generated by legacy DecisionEngine
        # ---------------------------------------------------------------------

        unsafe_prediction_keys = [

            "prediction",

            "predicted_label",

            "clinical_prediction",

            "clinical_confidence",
        ]

        for key in unsafe_prediction_keys:

            engine_decision.pop(
                key,
                None
            )

        # ---------------------------------------------------------------------
        # Final decision
        # ---------------------------------------------------------------------

        decision = dict(
            engine_decision
        )

        decision.update({

            "status":
                (
                    "completed"
                    if successful
                    else
                    "insufficient_evidence"
                ),

            "decision":
                decision_level,

            "decision_level":
                decision_level,

            "confidence":
                coordination_confidence,

            "coordination_confidence":
                coordination_confidence,

            "trust":
                weighted_trust,

            "coverage":
                coverage,

            "conflict_score":
                conflict_score,

            "num_agents":
                total,

            "num_valid_agents":
                valid_count,

            "request_additional_tests":
                request_additional_tests,

            "additional_test_reasons":
                reasons,

            "task_assessments":
                reasoning.get(
                    "task_assessments",
                    {}
                ),

            "heterogeneous_prediction_policy":
                (
                    "Predictions remain task-specific. "
                    "No cross-task numerical voting."
                ),
        })

        # ---------------------------------------------------------------------
        # Explanation
        # ---------------------------------------------------------------------

        decision["explanation"] = (

            f"{valid_count}/{total} agents provided valid evidence. "

            f"Coordination confidence="
            f"{coordination_confidence:.3f}, "

            f"trust="
            f"{weighted_trust:.3f}, "

            f"coverage="
            f"{coverage:.3f}, "

            f"conflict="
            f"{conflict_score:.3f}. "

            "Predictions from heterogeneous tasks "
            "were kept task-specific."
        )

        return decision

    # =========================================================================
    # ACTION
    # =========================================================================

    def _run_action(
        self,
        decision,
        reasoning
    ):
        """
        Generate an action from the coordination decision.
        """

        decision_for_action = dict(
            decision
        )

        decision_for_action[
            "decision_level"
        ] = decision.get(
            "decision_level",
            decision.get(
                "decision",
                "UNCERTAIN"
            )
        )

        try:

            action = (
                self.action_engine.generate(
                    decision_for_action
                )
            )

        except TypeError:

            try:

                action = (
                    self.action_engine.generate(
                        decision=
                            decision_for_action
                    )
                )

            except Exception as e:

                action = {

                    "status":
                        "action_generation_failed",

                    "actions": [
                        "Clinical review required."
                    ],

                    "error":
                        str(e),
                }

        except Exception as e:

            action = {

                "status":
                    "action_generation_failed",

                "actions": [
                    "Clinical review required."
                ],

                "error":
                    str(e),
            }

        if not isinstance(
            action,
            dict
        ):

            action = {

                "status":
                    "generated",

                "actions": [
                    str(action)
                ],
            }

        return action

    # =========================================================================
    # TASK SUMMARY
    # =========================================================================

    def _build_task_summary(
        self,
        results
    ):
        """
        Build a compact task-level summary.
        """

        groups = self._group_by_task(
            results
        )

        summary = {}

        for task_type, task_results in (
            groups.items()
        ):

            valid = [

                result

                for result in task_results

                if result.get("status")
                in (
                    "success",
                    "completed"
                )

                and result.get(
                    "prediction"
                ) is not None
            ]

            summary[
                task_type
            ] = {

                "num_agents":
                    len(task_results),

                "num_valid_agents":
                    len(valid),

                "predictions": [

                    result.get(
                        "prediction"
                    )

                    for result in valid
                ],

                "agents": [

                    result.get(
                        "agent_id"
                    )

                    for result in valid
                ],

                "mean_confidence": (

                    sum(

                        float(
                            result.get(
                                "confidence",
                                0.0
                            )
                        )

                        for result in valid
                    )

                    /

                    len(valid)

                    if valid

                    else 0.0
                ),

                "mean_trust": (

                    sum(

                        float(
                            result.get(
                                "trust",
                                0.0
                            )
                        )

                        for result in valid
                    )

                    /

                    len(valid)

                    if valid

                    else 0.0
                ),
            }

        return summary

    # =========================================================================
    # RUN
    # =========================================================================

    def run(
        self,
        patient_id=None,
        inputs=None,
        ground_truth=None
    ):
        """
        Execute the complete task-aware coordination pipeline.

        Parameters
        ----------
        patient_id : str, optional
            Patient identifier.

        inputs : dict
            Mapping:

                {
                    "CirrhosisAgent": {...},
                    "ClinicalReasoningAgent": {...}
                }

            Each agent receives ONLY its own input schema.

        ground_truth : optional
            Can be:

                {
                    "cirrhosis_classification": target,
                    "clinical_reasoning": target
                }

            or:

                {
                    "CirrhosisAgent": target
                }

            A single scalar is retained for backward compatibility,
            but should only be used when there is one corresponding task.
        """

        start_time = datetime.now()

        if inputs is None:

            inputs = {}

        if not isinstance(
            inputs,
            dict
        ):

            raise TypeError(
                "inputs must be a dictionary mapping "
                "agent_id -> input data."
            )

        # =====================================================================
        # STEP 1 — EXECUTE AGENTS
        # =====================================================================

        raw_results = []

        for agent_id, config in (
            self.agents.items()
        ):

            input_data = inputs.get(
                agent_id
            )

            result = self._execute_agent(

                agent_id,

                config,

                input_data
            )

            # =================================================================
            # STEP 2 — ADAPTIVE TRUST
            # =================================================================

            result = self._compute_trust(
                result
            )

            raw_results.append(
                result
            )

        # =====================================================================
        # STEP 3 — TASK-AWARE FUSION
        # =====================================================================

        try:

            fusion_result = (
                self.adaptive_fusion.fuse(
                    raw_results
                )
            )

        except Exception as e:

            fusion_result = {

                "status":
                    "fusion_error",

                "evidence":
                    [],

                "task_groups":
                    {},

                "weights":
                    {},

                "same_task_fusion":
                    {},

                "error":
                    str(e),
            }

        if not isinstance(
            fusion_result,
            dict
        ):

            fusion_result = {

                "status":
                    "fusion_error",

                "evidence":
                    [],

                "task_groups":
                    {},

                "weights":
                    {},

                "same_task_fusion":
                    {},

                "error":
                    "AdaptiveFusion returned a non-dictionary result.",
            }

        # =====================================================================
        # STEP 4 — CONFLICT DETECTION
        # =====================================================================

        conflict_error = None

        try:

            conflicts = (
                self.conflict_detector.detect(
                    raw_results
                )
            )

        except Exception as e:

            conflicts = []

            conflict_error = str(e)

        if not isinstance(
            conflicts,
            list
        ):

            conflicts = []

        # =====================================================================
        # STEP 5 — CONFLICT RESOLUTION
        # =====================================================================

        conflict_resolutions = (
            self._resolve_conflicts(

                raw_results,

                conflicts
            )
        )

        # =====================================================================
        # STEP 6 — TASK-AWARE REASONING
        # =====================================================================

        reasoning = self._run_reasoning(

            raw_results,

            fusion_result,

            conflicts,

            conflict_resolutions
        )

        # =====================================================================
        # STEP 7 — COORDINATION DECISION
        # =====================================================================

        decision = self._run_decision(

            raw_results,

            fusion_result,

            conflicts,

            reasoning
        )

        # =====================================================================
        # STEP 8 — ACTION
        # =====================================================================

        action = self._run_action(

            decision,

            reasoning
        )

        # =====================================================================
        # EXECUTION TIME
        # =====================================================================

        elapsed_ms = (

            datetime.now()
            -
            start_time

        ).total_seconds() * 1000.0

        # =====================================================================
        # TASK SUMMARY
        # =====================================================================

        task_summary = (
            self._build_task_summary(
                raw_results
            )
        )

        # =====================================================================
        # FINAL RESULT
        # =====================================================================

        final_result = {

            "status":
                "completed",

            "patient_id":
                patient_id,

            "timestamp":
                datetime.now().isoformat(),

            "execution_time_ms":
                elapsed_ms,

            "num_registered_agents":
                len(self.agents),

            "agents":
                raw_results,

            "task_summary":
                task_summary,

            "conflicts":
                conflicts,

            "conflict_resolution":
                conflict_resolutions,

            "fusion":
                fusion_result,

            "reasoning":
                reasoning,

            "decision":
                decision,

            "action":
                action,
        }

        # =====================================================================
        # ERROR INFORMATION
        # =====================================================================

        if conflict_error is not None:

            final_result[
                "conflict_detection_error"
            ] = conflict_error

        if fusion_result.get(
            "error"
        ):

            final_result[
                "fusion_error"
            ] = fusion_result[
                "error"
            ]

        # =====================================================================
        # FEEDBACK
        # =====================================================================

        if ground_truth is not None:

            try:

                final_result[
                    "feedback"
                ] = self.feedback(

                    raw_results,

                    ground_truth
                )

            except Exception as e:

                final_result[
                    "feedback"
                ] = {

                    "status":
                        "feedback_error",

                    "error":
                        str(e),
                }

        # =====================================================================
        # SAVE STATE
        # =====================================================================

        self.last_result = (
            final_result
        )

        self.execution_history.append(
            final_result
        )

        return final_result

    # =========================================================================
    # FEEDBACK
    # =========================================================================

    def feedback(
        self,
        agent_results,
        ground_truth
    ):
        """
        Update agent performance using task-specific ground truth.

        Recommended format:

            {
                "cirrhosis_classification": 1,
                "clinical_reasoning": 0
            }

        or:

            {
                "CirrhosisAgent": 1
            }

        Unrelated ground truth is not applied to an agent.
        """

        # ---------------------------------------------------------------------
        # Prefer FeedbackEngine when it supports the expected interface.
        # ---------------------------------------------------------------------

        try:

            result = self.feedback_engine.update(

                agent_results=
                    agent_results,

                ground_truth=
                    ground_truth
            )

            if isinstance(
                result,
                dict
            ):

                return result

        except Exception:

            pass

        # ---------------------------------------------------------------------
        # Safe fallback
        # ---------------------------------------------------------------------

        if not isinstance(
            agent_results,
            (list, tuple)
        ):

            agent_results = [
                agent_results
            ]

        feedback_results = []

        for result in agent_results:

            if not isinstance(
                result,
                dict
            ):

                continue

            prediction = result.get(
                "prediction"
            )

            if prediction is None:

                continue

            agent_id = result.get(

                "agent_id",

                result.get(
                    "agent",
                    "unknown"
                )
            )

            task_type = result.get(

                "task_type",

                "unknown"
            )

            # ---------------------------------------------------------------
            # Task-specific target
            # ---------------------------------------------------------------

            if isinstance(
                ground_truth,
                dict
            ):

                if task_type in ground_truth:

                    target = ground_truth[
                        task_type
                    ]

                elif agent_id in ground_truth:

                    target = ground_truth[
                        agent_id
                    ]

                else:

                    continue

            else:

                # -----------------------------------------------------------
                # Scalar target:
                # backward compatible, but only safe for one task.
                # -----------------------------------------------------------

                target = ground_truth

            correct = self._same_value(

                prediction,

                target
            )

            updated_trust = None

            try:

                updated_trust = (
                    self.trust_manager
                    .update_from_feedback(

                        agent_id=
                            agent_id,

                        correct=
                            correct
                    )
                )

            except Exception:

                pass

            feedback_results.append({

                "agent_id":
                    agent_id,

                "task_type":
                    task_type,

                "prediction":
                    prediction,

                "ground_truth":
                    target,

                "correct":
                    correct,

                "updated_historical_performance":
                    updated_trust,
            })

        return {

            "status":
                "completed",

            "results":
                feedback_results,
        }

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self):
        """
        Return coordinator status.
        """

        return {

            "status":
                "healthy",

            "name":
                self.name,

            "num_agents":
                len(self.agents),

            "agents":
                self.list_agents(),

            "has_last_result":
                self.last_result is not None,

            "execution_count":
                len(
                    self.execution_history
                ),
        }

    # =========================================================================
    # UTILITY — CLIP
    # =========================================================================

    @staticmethod
    def _clip(value):

        try:

            value = float(
                value
            )

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

    # =========================================================================
    # UTILITY — VALUE COMPARISON
    # =========================================================================

    @staticmethod
    def _same_value(
        a,
        b
    ):

        try:

            if a == b:

                return True

        except Exception:

            pass

        try:

            import numpy as np

            return bool(
                np.array_equal(
                    np.asarray(a),
                    np.asarray(b)
                )
            )

        except Exception:

            return str(a) == str(b)


# =============================================================================
# COMPATIBILITY ALIAS
# =============================================================================

LiverAICoordinator = LiverCoordinator
