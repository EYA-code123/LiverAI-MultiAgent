# =============================================================================
# LIVER AI — TASK-AWARE MULTI-AGENT COORDINATOR
# =============================================================================

from collections import defaultdict
from datetime import datetime

from coordinator.trust_manager import TrustManager
from coordinator.conflict_detector import ConflictDetector
from coordinator.conflict_resolver import ConflictResolver
from coordinator.adaptive_fusion import AdaptiveFusion
from coordinator.reasoning import EvidenceReasoner
from coordinator.decision import DecisionEngine
from coordinator.action import ActionEngine
from coordinator.feedback import FeedbackEngine


class LiverCoordinator:
    """
    Task-aware coordinator for the LiverAI multi-agent system.

    Important:
    - Different medical tasks are NOT merged into one global prediction.
    - Each agent keeps its own task-specific prediction.
    - Evidence is fused for coordination metrics only.
    - Conflicts are detected only between agents solving the same task.
    """

    # =========================================================================
    # INIT
    # =========================================================================

    def __init__(self, agents=None):

        self.agents = {}
        self.agent_metadata = {}

        # ---------------------------------------------------------------------
        # Coordination components
        # ---------------------------------------------------------------------

        self.trust_manager = TrustManager()

        self.conflict_detector = ConflictDetector()

        self.conflict_resolver = ConflictResolver(
            consensus_threshold=0.60
        )

        self.fusion = AdaptiveFusion(
            min_confidence=0.0,
            min_quality=0.0,
            use_trust=True
        )

        self.reasoner = EvidenceReasoner()

        self.decision_engine = DecisionEngine()

        self.action_engine = ActionEngine()

        self.feedback_engine = FeedbackEngine(
            self.trust_manager
        )

        # ---------------------------------------------------------------------
        # Optional initial agents
        # ---------------------------------------------------------------------

        if agents:

            self._load_agents(agents)

    # =========================================================================
    # LOAD AGENTS
    # =========================================================================

    def _load_agents(self, agents):

        if isinstance(agents, dict):

            for agent_id, agent_info in agents.items():

                if isinstance(agent_info, dict):

                    agent = agent_info.get("agent")

                    task_type = agent_info.get(
                        "task_type",
                        "unknown"
                    )

                    modality = agent_info.get(
                        "modality",
                        "unknown"
                    )

                else:

                    agent = agent_info
                    task_type = getattr(
                        agent,
                        "task_type",
                        "unknown"
                    )

                    modality = getattr(
                        agent,
                        "modality",
                        "unknown"
                    )

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
        task_type="unknown",
        modality="unknown"
    ):

        if agent is None:

            raise ValueError(
                f"Agent '{agent_id}' cannot be None."
            )

        self.agents[agent_id] = agent

        self.agent_metadata[agent_id] = {

            "task_type": task_type,

            "modality": modality
        }

        # Register agent in trust manager

        try:

            self.trust_manager.register_agent(
                agent_id
            )

        except TypeError:

            try:

                self.trust_manager.register_agent(
                    agent_id=agent_id
                )

            except Exception:

                pass

        return {
            "status": "registered",
            "agent_id": agent_id,
            "task_type": task_type,
            "modality": modality
        }

    # =========================================================================
    # UNREGISTER
    # =========================================================================

    def unregister_agent(self, agent_id):

        self.agents.pop(
            agent_id,
            None
        )

        self.agent_metadata.pop(
            agent_id,
            None
        )

        return {
            "status": "unregistered",
            "agent_id": agent_id
        }

    # =========================================================================
    # LIST AGENTS
    # =========================================================================

    def list_agents(self):

        output = []

        for agent_id in self.agents:

            metadata = self.agent_metadata.get(
                agent_id,
                {}
            )

            output.append({

                "agent_id": agent_id,

                "task_type": metadata.get(
                    "task_type",
                    "unknown"
                ),

                "modality": metadata.get(
                    "modality",
                    "unknown"
                )
            })

        return output

    # =========================================================================
    # GET AGENT
    # =========================================================================

    def get_agent(self, agent_id):

        return self.agents.get(
            agent_id
        )

    # =========================================================================
    # NORMALIZE RESULT
    # =========================================================================

    def _normalize_result(
        self,
        agent_id,
        result,
        task_type,
        modality
    ):

        if result is None:

            result = {}

        normalized = dict(result)

        normalized["agent_id"] = agent_id

        normalized["agent"] = normalized.get(
            "agent",
            agent_id
        )

        normalized["task_type"] = normalized.get(
            "task_type",
            task_type
        )

        normalized["modality"] = normalized.get(
            "modality",
            modality
        )

        normalized["status"] = normalized.get(
            "status",
            "success"
        )

        normalized["prediction"] = normalized.get(
            "prediction"
        )

        normalized["probability"] = normalized.get(
            "probability",
            normalized.get(
                "confidence"
            )
        )

        normalized["confidence"] = float(
            normalized.get(
                "confidence",
                0.0
            ) or 0.0
        )

        normalized["uncertainty"] = float(
            normalized.get(
                "uncertainty",
                1.0 - normalized["confidence"]
            ) or 0.0
        )

        normalized["quality"] = float(
            normalized.get(
                "quality",
                1.0
            ) or 0.0
        )

        normalized["missing_data_ratio"] = float(
            normalized.get(
                "missing_data_ratio",
                normalized.get(
                    "missing_ratio",
                    0.0
                )
            ) or 0.0
        )

        return normalized

    # =========================================================================
    # GROUP RESULTS BY TASK
    # =========================================================================

    def _group_by_task(self, results):

        grouped = defaultdict(list)

        for result in results:

            task_type = result.get(
                "task_type",
                "unknown"
            )

            grouped[task_type].append(
                result
            )

        return dict(grouped)

    # =========================================================================
    # TRUST
    # =========================================================================

    def _compute_trust(
        self,
        result,
        task_results
    ):

        agent_id = result.get(
            "agent_id"
        )

        confidence = float(
            result.get(
                "confidence",
                0.0
            ) or 0.0
        )

        quality = float(
            result.get(
                "quality",
                0.0
            ) or 0.0
        )

        uncertainty = float(
            result.get(
                "uncertainty",
                1.0
            ) or 1.0
        )

        # Agreement with other agents solving the same task

        valid_predictions = [

            r.get("prediction")

            for r in task_results

            if r.get("status") == "success"
            and r.get("prediction") is not None
        ]

        if len(valid_predictions) <= 1:

            agreement = 1.0

        else:

            prediction = result.get(
                "prediction"
            )

            matches = sum(

                1

                for p in valid_predictions

                if p == prediction
            )

            agreement = (
                matches /
                len(valid_predictions)
            )

        result["agreement"] = float(
            agreement
        )

        # ---------------------------------------------------------------------
        # Compute reliability
        # ---------------------------------------------------------------------

        try:

            reliability = (
                0.40 * confidence
                + 0.30 * quality
                + 0.20 * agreement
                + 0.10 * (1.0 - uncertainty)
            )

        except Exception:

            reliability = 0.0

        reliability = max(
            0.0,
            min(
                1.0,
                reliability
            )
        )

        # ---------------------------------------------------------------------
        # Trust manager
        # ---------------------------------------------------------------------

        try:

            trust = self.trust_manager.compute_trust(
                agent_id=agent_id,
                current_reliability=reliability
            )

        except TypeError:

            try:

                trust = self.trust_manager.compute_trust(
                    agent_id,
                    reliability
                )

            except Exception:

                trust = 0.5

        except Exception:

            trust = 0.5

        result["trust"] = float(
            trust
        )

        result["reliability"] = float(
            reliability
        )

        return result

    # =========================================================================
    # CONFLICT DETECTION
    # =========================================================================

    def _detect_conflicts(self, results):

        try:

            return self.conflict_detector.detect(
                results
            )

        except Exception:

            return []

    # =========================================================================
    # CONFLICT RESOLUTION
    # =========================================================================

    def _resolve_conflicts(
        self,
        results,
        conflicts
    ):

        grouped = self._group_by_task(
            results
        )

        resolutions = {}

        for task_type, task_results in grouped.items():

            task_conflicts = [

                conflict

                for conflict in conflicts

                if conflict.get(
                    "task_type"
                ) == task_type
            ]

            try:

                resolution = (
                    self.conflict_resolver.resolve(
                        task_type=task_type,
                        results=task_results,
                        conflicts=task_conflicts
                    )
                )

            except Exception as exc:

                resolution = {

                    "status": "failed",

                    "consensus": False,

                    "prediction": None,

                    "consensus_strength": 0.0,

                    "scores": {},

                    "reason": str(exc)
                }

            resolutions[task_type] = resolution

        return resolutions

    # =========================================================================
    # EVIDENCE GRAPH
    # =========================================================================

    def _build_evidence_graph(self, results):

        graph = []

        for i in range(
            len(results)
        ):

            for j in range(
                i + 1,
                len(results)
            ):

                first = results[i]

                second = results[j]

                first_task = first.get(
                    "task_type"
                )

                second_task = second.get(
                    "task_type"
                )

                # -------------------------------------------------------------
                # Different medical tasks
                # -------------------------------------------------------------

                if first_task != second_task:

                    relation = "complements"

                # -------------------------------------------------------------
                # Same task
                # -------------------------------------------------------------

                elif (
                    first.get("prediction")
                    ==
                    second.get("prediction")
                ):

                    relation = "supports"

                else:

                    relation = "conflicts"

                graph.append({

                    "agent_1": first.get(
                        "agent_id"
                    ),

                    "agent_2": second.get(
                        "agent_id"
                    ),

                    "task_1": first_task,

                    "task_2": second_task,

                    "relation": relation
                })

        return graph

    # =========================================================================
    # TASK DECISIONS
    # =========================================================================

    def _build_task_decisions(
        self,
        results,
        conflicts,
        resolutions
    ):

        grouped = self._group_by_task(
            results
        )

        task_decisions = {}

        task_actions = {}

        for task_type, task_results in grouped.items():

            valid_results = [

                result

                for result in task_results

                if result.get(
                    "status"
                ) == "success"
            ]

            resolution = resolutions.get(
                task_type,
                {}
            )

            prediction = resolution.get(
                "prediction"
            )

            if prediction is None and valid_results:

                prediction = valid_results[0].get(
                    "prediction"
                )

            # -------------------------------------------------------------
            # Aggregate confidence
            # -------------------------------------------------------------

            if valid_results:

                confidence_values = [

                    float(
                        result.get(
                            "confidence",
                            0.0
                        ) or 0.0
                    )

                    for result in valid_results
                ]

                confidence = sum(
                    confidence_values
                ) / len(
                    confidence_values
                )

            else:

                confidence = 0.0

            # -------------------------------------------------------------
            # Decision engine
            # -------------------------------------------------------------

            try:

                decision = self.decision_engine.decide(
                    task_results
                )

            except TypeError:

                try:

                    decision = self.decision_engine.decide(
                        results=task_results
                    )

                except Exception:

                    decision = {

                        "status": (
                            "insufficient_evidence"
                            if not valid_results
                            else "completed"
                        ),

                        "decision": (
                            "UNCERTAIN"
                            if confidence < 0.60
                            else "MODERATE"
                        ),

                        "prediction": prediction,

                        "confidence": confidence
                    }

            except Exception:

                decision = {

                    "status": (
                        "insufficient_evidence"
                        if not valid_results
                        else "completed"
                    ),

                    "decision": (
                        "UNCERTAIN"
                        if confidence < 0.60
                        else "MODERATE"
                    ),

                    "prediction": prediction,

                    "confidence": confidence
                }

            decision = dict(
                decision
            )

            # Normalize naming used by ActionEngine

            if "decision_level" not in decision:

                decision["decision_level"] = decision.get(
                    "decision",
                    "UNCERTAIN"
                )

            decision["prediction"] = prediction

            decision["confidence"] = confidence

            decision["task_type"] = task_type

            task_decisions[task_type] = decision

            # -------------------------------------------------------------
            # Actions
            # -------------------------------------------------------------

            try:

                action = self.action_engine.generate(
                    decision
                )

            except Exception:

                action = {

                    "status": "cautious",

                    "actions": [

                        "Clinical review recommended.",

                        "Consider additional evidence.",

                        "Do not use this automated output "
                        "as a standalone diagnosis."
                    ],

                    "referral": True,

                    "follow_up": True,

                    "additional_tests": True
                }

            task_actions[task_type] = action

        return (
            task_decisions,
            task_actions
        )

    # =========================================================================
    # COORDINATION SUMMARY
    # =========================================================================

    def _build_coordination_summary(
        self,
        results,
        conflicts
    ):

        valid_results = [

            result

            for result in results

            if result.get(
                "status"
            ) == "success"
        ]

        if valid_results:

            confidence_values = [

                float(
                    result.get(
                        "confidence",
                        0.0
                    ) or 0.0
                )

                for result in valid_results
            ]

            trust_values = [

                float(
                    result.get(
                        "trust",
                        0.0
                    ) or 0.0
                )

                for result in valid_results
            ]

            quality_values = [

                float(
                    result.get(
                        "quality",
                        0.0
                    ) or 0.0
                )

                for result in valid_results
            ]

            mean_confidence = (
                sum(confidence_values)
                /
                len(confidence_values)
            )

            mean_trust = (
                sum(trust_values)
                /
                len(trust_values)
            )

            mean_quality = (
                sum(quality_values)
                /
                len(quality_values)
            )

        else:

            mean_confidence = 0.0

            mean_trust = 0.0

            mean_quality = 0.0

        return {

            "num_agents": len(results),

            "num_valid_agents": len(
                valid_results
            ),

            "coverage": (
                len(valid_results)
                /
                len(results)
                if results
                else 0.0
            ),

            "mean_confidence": (
                mean_confidence
            ),

            "mean_trust": (
                mean_trust
            ),

            "mean_quality": (
                mean_quality
            ),

            "num_conflicts": len(
                conflicts
            ),

            "conflict_score": (
                len(conflicts)
                /
                max(
                    1,
                    len(results)
                )
            )
        }

    # =========================================================================
    # RUN
    # =========================================================================

    def run(
        self,
        patient_id=None,
        inputs=None,
        images=None,
        **kwargs
    ):

        # ---------------------------------------------------------------------
        # No agents
        # ---------------------------------------------------------------------

        if not self.agents:

            return {

                "status": "insufficient_evidence",

                "patient_id": patient_id,

                "agents": [],

                "error": "No agents registered."
            }

        inputs = (
            inputs
            if inputs is not None
            else {}
        )

        images = (
            images
            if images is not None
            else {}
        )

        results = []

        # =====================================================================
        # EXECUTE AGENTS
        # =====================================================================

        for agent_id, agent in self.agents.items():

            metadata = self.agent_metadata.get(
                agent_id,
                {}
            )

            task_type = metadata.get(
                "task_type",
                "unknown"
            )

            modality = metadata.get(
                "modality",
                "unknown"
            )

            # -----------------------------------------------------------------
            # Agent-specific input
            # -----------------------------------------------------------------

            if isinstance(inputs, dict):

                agent_input = inputs.get(
                    agent_id
                )

            else:

                agent_input = inputs

            # -----------------------------------------------------------------
            # Image-specific input
            # -----------------------------------------------------------------

            if isinstance(images, dict):

                agent_image = images.get(
                    agent_id
                )

            else:

                agent_image = images

            try:

                # =============================================================
                # FALLBACK
                # =============================================================

                if agent_input is None:

                    agent_input = kwargs.get(
                        "patient_data"
                    )

                # =============================================================
                # PREDICT
                # =============================================================

                if hasattr(
                    agent,
                    "predict"
                ):

                    # ---------------------------------------------------------
                    # 2D IMAGE AGENT
                    # ---------------------------------------------------------

                    if modality == "2D_image":

                        if agent_image is None:

                            raise ValueError(
                                f"No image provided for "
                                f"agent '{agent_id}'."
                            )

                        raw_result = agent.predict(
                            agent_image
                        )

                    # ---------------------------------------------------------
                    # ALL NON-IMAGE AGENTS
                    # ---------------------------------------------------------

                    else:

                        if agent_input is None:

                            raise ValueError(
                                f"No input provided for "
                                f"agent '{agent_id}'."
                            )

                        raw_result = agent.predict(
                            agent_input
                        )

                # =============================================================
                # ANALYZE
                # =============================================================

                elif hasattr(
                    agent,
                    "analyze"
                ):

                    raw_result = agent.analyze(
                        agent_input
                    )

                else:

                    raise AttributeError(

                        f"Agent '{agent_id}' has neither "
                        "'predict' nor 'analyze'."
                    )

                # =============================================================
                # NORMALIZE
                # =============================================================

                normalized = self._normalize_result(

                    agent_id=agent_id,

                    result=raw_result,

                    task_type=task_type,

                    modality=modality
                )

            except Exception as exc:

                normalized = {

                    "agent_id": agent_id,

                    "agent": agent_id,

                    "task_type": task_type,

                    "modality": modality,

                    "prediction": None,

                    "probability": None,

                    "confidence": 0.0,

                    "uncertainty": 1.0,

                    "quality": 0.0,

                    "missing_data_ratio": 1.0,

                    "status": "failed",

                    "error": str(exc),

                    "details": {}
                }

            results.append(
                normalized
            )

        # =====================================================================
        # TRUST
        # =====================================================================

        grouped = self._group_by_task(
            results
        )

        for result in results:

            task_results = grouped.get(
                result.get(
                    "task_type"
                ),
                []
            )

            if result.get(
                "status"
            ) == "success":

                self._compute_trust(

                    result,

                    task_results
                )

            else:

                result["trust"] = 0.0

                result["agreement"] = 0.0

        # =====================================================================
        # ADAPTIVE FUSION
        # =====================================================================

        try:

            fusion_result = self.fusion.fuse(
                results
            )

        except Exception as exc:

            fusion_result = {

                "status": "failed",

                "error": str(exc)
            }

        # =====================================================================
        # CONFLICTS
        # =====================================================================

        conflicts = self._detect_conflicts(
            results
        )

        resolutions = self._resolve_conflicts(

            results,

            conflicts
        )

        # =====================================================================
        # EVIDENCE GRAPH
        # =====================================================================

        evidence_graph = self._build_evidence_graph(
            results
        )

        # =====================================================================
        # TASK DECISIONS + ACTIONS
        # =====================================================================

        (
            task_decisions,
            task_actions
        ) = self._build_task_decisions(

            results,

            conflicts,

            resolutions
        )

        # =====================================================================
        # REASONING
        # =====================================================================

        reasoning_result = {

            "status": (
                "completed"
                if results
                else "insufficient_evidence"
            ),

            "task_assessments": {

                task: {

                    "prediction": resolutions.get(
                        task,
                        {}
                    ).get(
                        "prediction"
                    ),

                    "confidence": resolutions.get(
                        task,
                        {}
                    ).get(
                        "consensus_strength",
                        0.0
                    ),

                    "resolution": resolutions.get(
                        task,
                        {}
                    )
                }

                for task in grouped
            },

            "evidence_graph": evidence_graph,

            "explanation": (

                "Evidence was evaluated separately "
                "for each medical task. "
                "Different tasks are complementary "
                "and are not merged into one prediction."
            )
        }

        # =====================================================================
        # GLOBAL DECISION
        # =====================================================================

        coordination = (
            self._build_coordination_summary(

                results,

                conflicts
            )
        )

        decision_result = {

            "status": (

                "completed"
                if results
                else "insufficient_evidence"
            ),

            "task_decisions": task_decisions,

            "coordination": coordination,

            # IMPORTANT:
            # No global medical prediction.

            "prediction": None,

            "explanation": (

                "Decisions are task-specific. "
                "No global medical class is inferred "
                "from heterogeneous agents."
            )
        }

        # =====================================================================
        # FINAL RESULT
        # =====================================================================

        return {

            "status": "completed",

            "patient_id": patient_id,

            "agents": results,

            "task_summary": {

                task: {

                    "num_agents": len(
                        task_results
                    ),

                    "prediction": resolutions.get(
                        task,
                        {}
                    ).get(
                        "prediction"
                    ),

                    "conflict": any(

                        c.get(
                            "task_type"
                        ) == task

                        for c in conflicts
                    )
                }

                for task, task_results
                in grouped.items()
            },

            "conflicts": conflicts,

            "conflict_resolution": resolutions,

            "fusion": fusion_result,

            "reasoning": reasoning_result,

            "decision": decision_result,

            "action": {

                "task_actions": task_actions
            },

            "timestamp": datetime.utcnow().isoformat()
        }

    # =========================================================================
    # FEEDBACK
    # =========================================================================

    def update_feedback(
        self,
        agent_results,
        ground_truths
    ):

        if not agent_results:

            return {
                "status": "no_results"
            }

        # ---------------------------------------------------------------------
        # Group by task
        # ---------------------------------------------------------------------

        grouped = self._group_by_task(
            agent_results
        )

        feedback_results = {}

        for task_type, results in grouped.items():

            # Ground truth may be:
            # {
            #     "cirrhosis_classification": 1,
            #     "fibrosis_classification": 0
            # }

            if isinstance(
                ground_truths,
                dict
            ):

                ground_truth = ground_truths.get(
                    task_type
                )

            else:

                ground_truth = ground_truths

            if ground_truth is None:

                continue

            try:

                feedback = (
                    self.feedback_engine.update(
                        results,
                        ground_truth
                    )
                )

            except Exception as exc:

                feedback = {

                    "status": "failed",

                    "error": str(exc)
                }

            feedback_results[task_type] = feedback

        return {

            "status": "completed",

            "feedback": feedback_results
        }

    # =========================================================================
    # FEEDBACK ALIAS
    # =========================================================================

    def feedback(
        self,
        agent_results,
        ground_truths
    ):

        return self.update_feedback(
            agent_results,
            ground_truths
        )

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self):

        agents_status = []

        for agent_id, agent in self.agents.items():

            metadata = self.agent_metadata.get(
                agent_id,
                {}
            )

            agents_status.append({

                "agent_id": agent_id,

                "loaded": agent is not None,

                "has_predict": hasattr(
                    agent,
                    "predict"
                ),

                "has_analyze": hasattr(
                    agent,
                    "analyze"
                ),

                "task_type": metadata.get(
                    "task_type",
                    "unknown"
                ),

                "modality": metadata.get(
                    "modality",
                    "unknown"
                )
            })

        return {

            "status": (
                "healthy"
                if agents_status
                else "empty"
            ),

            "num_agents": len(
                self.agents
            ),

            "agents": agents_status
        }


# =============================================================================
# BACKWARD-COMPATIBILITY ALIAS
# =============================================================================

LiverAICoordinator = LiverCoordinator
