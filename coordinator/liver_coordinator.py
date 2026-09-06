# =============================================================================
# LiverAI-MultiAgent
# LIVER AI COORDINATOR
# TASK-AWARE ADAPTIVE COORDINATION
# =============================================================================

from collections import defaultdict
from datetime import datetime
import traceback


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


class LiverCoordinator:
    """
    Central coordination engine for LiverAI-MultiAgent.

    Architecture:

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
        Task-Aware Evidence Fusion
             |
             v
        Same-Task Conflict Detection
             |
             v
        Conflict Resolution
             |
             v
        Evidence Reasoning
             |
             v
        Decision Intelligence
             |
             v
        Action Generation

    IMPORTANT
    ---------
    Agents performing different tasks are NOT directly compared.

    Example:

        CirrhosisAgent
            task = cirrhosis_classification

        ClinicalReasoningAgent
            task = clinical_reasoning

    Their numerical predictions are not assumed to represent the
    same target.
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
        # State
        # ---------------------------------------------------------------------

        self.last_result = None

        self.execution_history = []

        # ---------------------------------------------------------------------
        # Optional initial agents
        # ---------------------------------------------------------------------

        if agents is not None:

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

                    if isinstance(item, dict):

                        self.register_agent(
                            agent_id=item.get(
                                "agent_id",
                                item.get("agent", "unknown")
                            ),
                            agent=item.get("agent"),
                            task_type=item.get(
                                "task_type",
                                "unknown"
                            ),
                            modality=item.get(
                                "modality",
                                "unknown"
                            )
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

        Parameters
        ----------
        agent_id : str
            Unique identifier.

        agent : object
            Agent instance exposing predict() or analyze().

        task_type : str
            Semantic task performed by the agent.

        modality : str
            Input modality.
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

        self.agents[agent_id] = {
            "agent_id": agent_id,
            "agent": agent,
            "task_type": str(task_type),
            "modality": str(modality),
        }

        return {
            "status": "registered",
            "agent_id": agent_id,
            "task_type": str(task_type),
            "modality": str(modality),
        }

    # =========================================================================
    # UNREGISTER
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

        return {
            agent_id: {
                "task_type": config["task_type"],
                "modality": config["modality"],
                "agent": type(
                    config["agent"]
                ).__name__,
            }
            for agent_id, config in self.agents.items()
        }

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
        Convert heterogeneous agent outputs into one common dictionary.

        This function does not alter the prediction semantics.
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

        result["status"] = result.get(
            "status",
            "success"
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
            "confidence",
            None
        )

        if confidence is None:

            probability = result.get(
                "probability",
                None
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

        result["quality"] = self._clip(
            result.get(
                "quality",
                1.0
                if result["status"]
                in ("success", "completed")
                else 0.0
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
        # Stability / utility
        # ---------------------------------------------------------------------

        result["stability"] = self._clip(
            result.get(
                "stability",
                0.5
            )
        )

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
        Compute patient-specific adaptive trust.

        Trust is computed independently for each agent.
        """

        try:

            trust = self.trust_manager.compute_trust(

                agent_id=result["agent_id"],

                confidence=result["confidence"],

                quality=result["quality"],

                uncertainty=result["uncertainty"],

                missing_data_ratio=(
                    result["missing_data_ratio"]
                )
            )

        except TypeError:

            # Compatibility with older TrustManager signatures.
            trust = self.trust_manager.compute_trust(

                result["agent_id"],

                result["confidence"],

                result["quality"],

                result["uncertainty"],

                result["missing_data_ratio"]
            )

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
        Safely execute one registered agent.
        """

        task_type = config["task_type"]

        modality = config["modality"]

        agent = config["agent"]

        # ---------------------------------------------------------------------
        # No input
        # ---------------------------------------------------------------------

        if input_data is None:

            return self._normalize_result(

                {
                    "status": "no_input",
                    "prediction": None,
                    "confidence": 0.0,
                    "uncertainty": 1.0,
                    "quality": 0.0,
                    "missing_data_ratio": 1.0,
                    "error": "No input provided.",
                },

                agent_id,
                task_type,
                modality
            )

        # ---------------------------------------------------------------------
        # Execute
        # ---------------------------------------------------------------------

        start = datetime.now()

        try:

            if hasattr(agent, "predict"):

                result = agent.predict(
                    input_data
                )

            elif hasattr(agent, "analyze"):

                result = agent.analyze(
                    input_data
                )

            else:

                raise TypeError(
                    f"Agent '{agent_id}' has no "
                    "predict() or analyze() method."
                )

            elapsed = (
                datetime.now() - start
            ).total_seconds() * 1000.0

            result = self._normalize_result(

                result,

                agent_id,

                task_type,

                modality
            )

            if result.get("latency_ms", 0.0) <= 0:

                result["latency_ms"] = elapsed

            return result

        except Exception as e:

            elapsed = (
                datetime.now() - start
            ).total_seconds() * 1000.0

            return self._normalize_result(

                {
                    "status": "error",
                    "prediction": None,
                    "confidence": 0.0,
                    "uncertainty": 1.0,
                    "quality": 0.0,
                    "missing_data_ratio": 1.0,
                    "latency_ms": elapsed,
                    "error": (
                        f"{type(e).__name__}: {e}"
                    ),
                },

                agent_id,
                task_type,
                modality
            )

    # =========================================================================
    # TASK GROUPING
    # =========================================================================

    @staticmethod
    def _group_by_task(results):

        groups = defaultdict(list)

        for result in results:

            if result is None:
                continue

            task_type = result.get(
                "task_type",
                "unknown"
            )

            groups[
                str(task_type)
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
        Resolve conflicts only within their task.

        Heterogeneous tasks remain independent.
        """

        if not conflicts:

            return {}

        task_groups = self._group_by_task(
            results
        )

        resolutions = {}

        conflict_tasks = set(
            conflict.get(
                "task_type",
                "unknown"
            )
            for conflict in conflicts
        )

        for task_type in conflict_tasks:

            task_results = task_groups.get(
                task_type,
                []
            )

            if not task_results:

                continue

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
                            conflicts=[
                                c for c in conflicts
                                if c.get(
                                    "task_type"
                                ) == task_type
                            ],
                            results=task_results
                        )
                    )

                except Exception as e:

                    resolution = {
                        "status": "resolution_failed",
                        "task_type": task_type,
                        "error": str(e),
                    }

            except Exception as e:

                resolution = {
                    "status": "resolution_failed",
                    "task_type": task_type,
                    "error": str(e),
                }

            resolutions[
                task_type
            ] = resolution

        return resolutions

    # =========================================================================
    # TASK-AWARE REASONING
    # =========================================================================

    def _build_task_assessments(
        self,
        results,
        fusion_result,
        conflict_resolutions
    ):
        """
        Build independent assessments for every task.

        No cross-task numerical voting occurs here.
        """

        groups = self._group_by_task(
            results
        )

        task_assessments = {}

        fusion_tasks = (
            fusion_result.get(
                "same_task_fusion",
                {}
            )
            if isinstance(
                fusion_result,
                dict
            )
            else {}
        )

        for task_type, items in groups.items():

            valid = [
                item
                for item in items
                if item.get("status")
                in ("success", "completed")
                and item.get("prediction") is not None
            ]

            if not valid:

                task_assessments[
                    task_type
                ] = {
                    "task_type": task_type,
                    "status": "no_valid_evidence",
                    "prediction": None,
                    "confidence": 0.0,
                    "uncertainty": 1.0,
                    "agents": [],
                }

                continue

            # ---------------------------------------------------------------
            # If several agents solve the same task and fusion is available
            # ---------------------------------------------------------------

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

                confidence = self._clip(
                    fused.get(
                        "confidence",
                        0.0
                    )
                )

            else:

                # -----------------------------------------------------------
                # One task / one agent
                # -----------------------------------------------------------

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

            uncertainty = (
                1.0 - confidence
            )

            task_assessments[
                task_type
            ] = {
                "task_type": task_type,
                "status": "completed",
                "prediction": prediction,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "num_agents": len(valid),
                "supporting_agents": [
                    item["agent_id"]
                    for item in valid
                ],
                "conflict_resolution": (
                    conflict_resolutions.get(
                        task_type
                    )
                ),
            }

        return task_assessments

    # =========================================================================
    # REASONING
    # =========================================================================

    def _run_reasoning(
        self,
        results,
        fusion_result,
        conflicts,
        conflict_resolutions
    ):
        """
        Produce task-aware reasoning.

        IMPORTANT:
        ClinicalReasoningAgent is treated as an evidence source,
        not automatically as a universal final predictor.
        """

        task_assessments = (
            self._build_task_assessments(
                results,
                fusion_result,
                conflict_resolutions
            )
        )

        # ---------------------------------------------------------------------
        # Build evidence graph
        # ---------------------------------------------------------------------

        nodes = []

        for result in results:

            if result.get("status") not in (
                "success",
                "completed"
            ):

                continue

            node_id = (
                f"{result['agent_id']}:"
                f"{result.get('prediction')}"
            )

            nodes.append({

                "id": node_id,

                "agent":
                    result["agent_id"],

                "task":
                    result["task_type"],

                "prediction":
                    result.get("prediction"),

                "confidence":
                    result.get("confidence", 0.0),

                "trust":
                    result.get("trust", 0.0),

                "quality":
                    result.get("quality", 0.0),
            })

        # ---------------------------------------------------------------------
        # Edges
        # ---------------------------------------------------------------------

        edges = []

        for i in range(len(nodes)):

            for j in range(i + 1, len(nodes)):

                a = nodes[i]
                b = nodes[j]

                # Same task
                if a["task"] == b["task"]:

                    if a["prediction"] == b["prediction"]:

                        relation = "supports"

                    else:

                        relation = "conflicts"

                # Different tasks
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
        # Try EvidenceReasoner
        # ---------------------------------------------------------------------

        reasoning = None

        try:

            reasoning = self.evidence_reasoner.synthesize(
                results,
                fusion_result,
                conflicts,
                conflict_resolutions
            )

        except TypeError:

            try:

                reasoning = self.evidence_reasoner.synthesize(
                    results
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
        # Remove unsafe global prediction
        # ---------------------------------------------------------------------

        # We deliberately do NOT create:
        #
        #     prediction = 1
        #
        # from heterogeneous tasks.
        #
        # Each task keeps its own prediction.

        reasoning["status"] = "completed"

        reasoning["task_assessments"] = (
            task_assessments
        )

        reasoning["evidence_graph"] = {

            "nodes": nodes,

            "edges": edges,
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
                f"supported by={agents}"
            )

        reasoning["explanation"] = (
            "Task-aware evidence assessment. "
            +
            " | ".join(explanations)
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
        Generate a global coordination decision without pretending that
        heterogeneous predictions are the same target.
        """

        successful = [
            result
            for result in results
            if result.get("status")
            in ("success", "completed")
            and result.get("prediction") is not None
        ]

        total = len(results)

        valid_count = len(successful)

        coverage = (
            valid_count / total
            if total > 0
            else 0.0
        )

        # ---------------------------------------------------------------------
        # Weighted confidence / trust
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
                        result.get(
                            "confidence",
                            0.0
                        ) * weight

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
                        result.get(
                            "trust",
                            0.0
                        ) * weight

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

        # ---------------------------------------------------------------------
        # Conflict score
        # ---------------------------------------------------------------------

        if conflicts:

            conflict_strengths = [

                float(
                    conflict.get(
                        "conflict_strength",
                        0.0
                    )
                )

                for conflict in conflicts
            ]

            conflict_score = (
                sum(conflict_strengths)
                /
                len(conflict_strengths)
            )

        else:

            conflict_score = 0.0

        # ---------------------------------------------------------------------
        # Global coordination confidence
        # ---------------------------------------------------------------------

        coordination_confidence = (

            0.35 * weighted_confidence

            +

            0.25 * weighted_trust

            +

            0.20 * coverage

            +

            0.20 * (
                1.0 - conflict_score
            )
        )

        coordination_confidence = self._clip(
            coordination_confidence
        )

        # ---------------------------------------------------------------------
        # Additional tests
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
                "Strong conflict detected between "
                "agents performing compatible tasks."
            )

        # ---------------------------------------------------------------------
        # Decision level
        # ---------------------------------------------------------------------

        if not successful:

            decision_level = "INSUFFICIENT_EVIDENCE"

        elif request_additional_tests:

            decision_level = "UNCERTAIN"

        elif coordination_confidence >= 0.80:

            decision_level = "HIGH"

        elif coordination_confidence >= 0.55:

            decision_level = "MODERATE"

        else:

            decision_level = "UNCERTAIN"

        # ---------------------------------------------------------------------
        # Use DecisionEngine when possible
        # ---------------------------------------------------------------------

        engine_decision = {}

        try:

            engine_decision = (
                self.decision_engine.decide(
                    agent_results=results,
                    conflicts=conflicts,
                    fused_results=fusion_result,
                    clinical_reasoning=reasoning
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
        # IMPORTANT:
        # Do not allow the old engine to inject a heterogeneous prediction.
        # ---------------------------------------------------------------------

        engine_decision.pop(
            "prediction",
            None
        )

        engine_decision.pop(
            "clinical_prediction",
            None
        )

        engine_decision.pop(
            "clinical_confidence",
            None
        )

        # ---------------------------------------------------------------------
        # Final coordinated decision
        # ---------------------------------------------------------------------

        decision = dict(
            engine_decision
        )

        decision.update({

            "status":
                "completed"
                if successful
                else "insufficient_evidence",

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
        })

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

        # ActionEngine expects decision_level.
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

            action = self.action_engine.generate(
                decision_for_action
            )

        except TypeError:

            try:

                action = self.action_engine.generate(
                    decision=decision_for_action
                )

            except Exception as e:

                action = {
                    "status": "action_generation_failed",
                    "actions": [
                        "Clinical review required."
                    ],
                    "error": str(e),
                }

        except Exception as e:

            action = {
                "status": "action_generation_failed",
                "actions": [
                    "Clinical review required."
                ],
                "error": str(e),
            }

        if not isinstance(
            action,
            dict
        ):

            action = {
                "status": "generated",
                "actions": [str(action)]
            }

        return action

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
        Execute the complete coordination pipeline.

        Parameters
        ----------
        patient_id : str
            Patient identifier.

        inputs : dict
            Mapping:

                {
                    "CirrhosisAgent": {...},
                    "ClinicalReasoningAgent": {...}
                }

            Each agent receives ONLY its own expected input schema.

        ground_truth : optional
            Ground truth used for feedback.
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

        # ---------------------------------------------------------------------
        # STEP 1 — EXECUTION
        # ---------------------------------------------------------------------

        raw_results = []

        for agent_id, config in self.agents.items():

            input_data = inputs.get(
                agent_id
            )

            result = self._execute_agent(

                agent_id,

                config,

                input_data
            )

            # -------------------------------------------------------------
            # STEP 2 — ADAPTIVE TRUST
            # -------------------------------------------------------------

            result = self._compute_trust(
                result
            )

            raw_results.append(
                result
            )

        # ---------------------------------------------------------------------
        # STEP 3 — TASK-AWARE FUSION
        # ---------------------------------------------------------------------

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

                "evidence": [],

                "task_groups": {},

                "weights": {},

                "same_task_fusion": {},

                "error":
                    str(e),
            }

        # ---------------------------------------------------------------------
        # STEP 4 — CONFLICT DETECTION
        # ---------------------------------------------------------------------

        try:

            conflicts = (
                self.conflict_detector.detect(
                    raw_results
                )
            )

        except Exception as e:

            conflicts = []

            conflict_error = str(e)

        else:

            conflict_error = None

        # ---------------------------------------------------------------------
        # STEP 5 — CONFLICT RESOLUTION
        # ---------------------------------------------------------------------

        conflict_resolutions = (
            self._resolve_conflicts(
                raw_results,
                conflicts
            )
        )

        # ---------------------------------------------------------------------
        # STEP 6 — TASK-AWARE REASONING
        # ---------------------------------------------------------------------

        reasoning = self._run_reasoning(

            raw_results,

            fusion_result,

            conflicts,

            conflict_resolutions
        )

        # ---------------------------------------------------------------------
        # STEP 7 — DECISION
        # ---------------------------------------------------------------------

        decision = self._run_decision(

            raw_results,

            fusion_result,

            conflicts,

            reasoning
        )

        # ---------------------------------------------------------------------
        # STEP 8 — ACTION
        # ---------------------------------------------------------------------

        action = self._run_action(

            decision,

            reasoning
        )

        # ---------------------------------------------------------------------
        # EXECUTION TIME
        # ---------------------------------------------------------------------

        elapsed_ms = (
            datetime.now() - start_time
        ).total_seconds() * 1000.0

        # ---------------------------------------------------------------------
        # TASK SUMMARY
        # ---------------------------------------------------------------------

        task_groups = self._group_by_task(
            raw_results
        )

        task_summary = {}

        for task_type, task_results in (
            task_groups.items()
        ):

            valid = [
                r for r in task_results
                if r.get("status")
                in ("success", "completed")
                and r.get("prediction") is not None
            ]

            task_summary[
                task_type
            ] = {

                "num_agents":
                    len(task_results),

                "num_valid_agents":
                    len(valid),

                "predictions": [
                    r.get("prediction")
                    for r in valid
                ],

                "agents": [
                    r.get("agent_id")
                    for r in valid
                ],

                "mean_confidence": (
                    sum(
                        float(
                            r.get(
                                "confidence",
                                0.0
                            )
                        )
                        for r in valid
                    )
                    /
                    len(valid)
                    if valid
                    else 0.0
                ),

                "mean_trust": (
                    sum(
                        float(
                            r.get(
                                "trust",
                                0.0
                            )
                        )
                        for r in valid
                    )
                    /
                    len(valid)
                    if valid
                    else 0.0
                ),
            }

        # ---------------------------------------------------------------------
        # FINAL RESULT
        # ---------------------------------------------------------------------

        final_result = {

            "status":
                "completed",

            "patient_id":
                patient_id,

            "timestamp":
                datetime.now().isoformat(),

            "execution_time_ms":
                elapsed_ms,

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

        # ---------------------------------------------------------------------
        # FEEDBACK
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # SAVE STATE
        # ---------------------------------------------------------------------

        self.last_result = final_result

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
        Update agent performance when ground truth is available.

        IMPORTANT:
        Feedback should only be applied when the ground truth belongs
        to the corresponding task. A single global ground truth must
        not automatically be applied to heterogeneous agents.
        """

        feedback_results = []

        if not isinstance(
            agent_results,
            (list, tuple)
        ):

            agent_results = [
                agent_results
            ]

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

            # -------------------------------------------------------------
            # Ground truth can be task-specific
            # -------------------------------------------------------------

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

                    # Do not apply unrelated truth.
                    continue

            else:

                # Backward-compatible single target.
                target = ground_truth

            correct = self._same_value(
                prediction,
                target
            )

            try:

                updated_trust = (
                    self.trust_manager
                    .update_from_feedback(
                        agent_id=agent_id,
                        correct=correct
                    )
                )

            except Exception:

                updated_trust = None

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

    # -------------------------------------------------------------------------

    @staticmethod
    def _same_value(a, b):

        if a == b:

            return True

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
