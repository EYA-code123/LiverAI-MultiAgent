# =============================================================================
# LIVER AI — LIVER COORDINATOR
# =============================================================================
#
# Task-aware multi-agent coordinator for heterogeneous liver AI agents.
#
# Supported modalities:
#   - clinical_tabular
#   - 2D_image
#   - 3D_CT
#
# The coordinator:
#   1. Executes registered agents
#   2. Normalizes heterogeneous outputs
#   3. Computes trust
#   4. Detects conflicts within the same task
#   5. Resolves task-specific conflicts
#   6. Builds task-aware evidence
#   7. Performs heterogeneous evidence fusion
#   8. Produces task-specific decisions and actions
#
# IMPORTANT:
# Different medical tasks are NOT merged into one global prediction.
#
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

    # =========================================================================
    # INITIALIZATION
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
                    task_type = "unknown"
                    modality = "unknown"

                if agent is not None:

                    self.register_agent(
                        agent_id=agent_id,
                        agent=agent,
                        task_type=task_type,
                        modality=modality
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

        if agent is None:
            raise ValueError(
                f"Agent '{agent_id}' cannot be None."
            )

        self.agents[agent_id] = agent

        self.agent_metadata[agent_id] = {
            "task_type": task_type,
            "modality": modality
        }

        # Register initial trust
        try:

            self.trust_manager.register_agent(
                agent_id=agent_id,
                performance=0.5
            )

        except Exception:

            # Avoid breaking registration if the trust manager
            # already contains this agent.
            pass

        return {
            "status": "registered",
            "agent_id": agent_id,
            "task_type": task_type,
            "modality": modality
        }

    # =========================================================================
    # UNREGISTER AGENT
    # =========================================================================

    def unregister_agent(self, agent_id):

        if agent_id in self.agents:
            del self.agents[agent_id]

        if agent_id in self.agent_metadata:
            del self.agent_metadata[agent_id]

        return {
            "status": "unregistered",
            "agent_id": agent_id
        }

    # =========================================================================
    # LIST AGENTS
    # =========================================================================

    def list_agents(self):

        return list(self.agents.keys())

    # =========================================================================
    # GET AGENT
    # =========================================================================

    def get_agent(self, agent_id):

        return self.agents.get(agent_id)

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

            return {
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
                "error": "Agent returned None.",
                "details": {}
            }

        if not isinstance(result, dict):

            result = {
                "prediction": result
            }

        prediction = result.get(
            "prediction"
        )

        probability = result.get(
            "probability"
        )

        confidence = result.get(
            "confidence",
            probability if probability is not None else 0.0
        )

        uncertainty = result.get(
            "uncertainty",
            1.0 - float(confidence)
        )

        quality = result.get(
            "quality",
            1.0
        )

        missing_ratio = result.get(
            "missing_data_ratio",
            result.get(
                "missing_ratio",
                0.0
            )
        )

        status = result.get(
            "status",
            "success"
        )

        normalized = {
            "agent_id": agent_id,

            "agent": result.get(
                "agent",
                agent_id
            ),

            "task_type": result.get(
                "task_type",
                task_type
            ),

            "modality": result.get(
                "modality",
                modality
            ),

            "prediction": (
                str(prediction)
                if prediction is not None
                else None
            ),

            "probability": probability,

            "confidence": float(
                confidence
            ),

            "uncertainty": float(
                uncertainty
            ),

            "quality": float(
                quality
            ),

            "missing_data_ratio": float(
                missing_ratio
            ),

            "status": status,

            "error": result.get(
                "error"
            ),

            "details": result
        }

        return normalized

    # =========================================================================
    # GROUP RESULTS BY TASK
    # =========================================================================

    def _group_by_task(self, results):

        grouped = defaultdict(list)

        for result in results:

            task = result.get(
                "task_type",
                "unknown"
            )

            grouped[task].append(
                result
            )

        return dict(grouped)

    # =========================================================================
    # COMPUTE TRUST
    # =========================================================================

    def _compute_trust(
        self,
        result,
        task_results
    ):

        confidence = float(
            result.get(
                "confidence",
                0.0
            )
        )

        quality = float(
            result.get(
                "quality",
                0.0
            )
        )

        uncertainty = float(
            result.get(
                "uncertainty",
                1.0
            )
        )

        try:

            trust = self.trust_manager.compute_trust(
                agent_id=result["agent_id"],
                confidence=confidence,
                quality=quality,
                uncertainty=uncertainty
            )

        except TypeError:

            try:

                trust = self.trust_manager.compute_trust(
                    result["agent_id"],
                    confidence,
                    quality,
                    uncertainty
                )

            except Exception:

                trust = (
                    0.5 * confidence
                    + 0.3 * quality
                    + 0.2 * (1.0 - uncertainty)
                )

        except Exception:

            trust = (
                0.5 * confidence
                + 0.3 * quality
                + 0.2 * (1.0 - uncertainty)
            )

        result["trust"] = float(
            max(
                0.0,
                min(
                    1.0,
                    trust
                )
            )
        )

        # ---------------------------------------------------------------------
        # Agreement with other agents of the SAME task
        # ---------------------------------------------------------------------

        same_task = [
            r for r in task_results
            if r.get("agent_id") != result.get("agent_id")
            and r.get("status") == "success"
            and r.get("prediction") is not None
        ]

        if not same_task:

            result["agreement"] = 0.5

        else:

            agreements = [
                1.0
                if r.get("prediction") == result.get("prediction")
                else 0.0
                for r in same_task
            ]

            result["agreement"] = (
                sum(agreements)
                / len(agreements)
            )

    # =========================================================================
    # DETECT CONFLICTS
    # =========================================================================

    def _detect_conflicts(self, results):

        try:

            return self.conflict_detector.detect(
                results
            )

        except Exception:

            conflicts = []

            grouped = self._group_by_task(
                results
            )

            for task_type, task_results in grouped.items():

                valid = [
                    r for r in task_results
                    if r.get("status") == "success"
                    and r.get("prediction") is not None
                ]

                for i in range(len(valid)):

                    for j in range(i + 1, len(valid)):

                        r1 = valid[i]
                        r2 = valid[j]

                        if (
                            r1["prediction"]
                            !=
                            r2["prediction"]
                        ):

                            conflicts.append({

                                "task_type": task_type,

                                "agent_1":
                                    r1["agent_id"],

                                "prediction_1":
                                    r1["prediction"],

                                "confidence_1":
                                    r1["confidence"],

                                "agent_2":
                                    r2["agent_id"],

                                "prediction_2":
                                    r2["prediction"],

                                "confidence_2":
                                    r2["confidence"]
                            })

            return conflicts

    # =========================================================================
    # RESOLVE CONFLICTS
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
                c for c in conflicts
                if c.get("task_type") == task_type
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
                    "status": "error",
                    "consensus": False,
                    "prediction": None,
                    "consensus_strength": 0.0,
                    "scores": {},
                    "reason": str(exc)
                }

            resolutions[task_type] = resolution

        return resolutions

    # =========================================================================
    # TASK-AWARE EVIDENCE GRAPH
    # =========================================================================

    def _build_evidence_graph(self, results):

        graph = []

        valid_results = [
            r for r in results
            if r.get("status") == "success"
        ]

        for i in range(len(valid_results)):

            for j in range(i + 1, len(valid_results)):

                r1 = valid_results[i]
                r2 = valid_results[j]

                task1 = r1.get(
                    "task_type"
                )

                task2 = r2.get(
                    "task_type"
                )

                # -------------------------------------------------------------
                # Different tasks = complementary evidence
                # -------------------------------------------------------------

                if task1 != task2:

                    relation = "complements"

                # -------------------------------------------------------------
                # Same task / same prediction = support
                # -------------------------------------------------------------

                elif (
                    r1.get("prediction")
                    ==
                    r2.get("prediction")
                ):

                    relation = "supports"

                # -------------------------------------------------------------
                # Same task / different prediction = conflict
                # -------------------------------------------------------------

                else:

                    relation = "conflicts"

                graph.append({

                    "agent_1":
                        r1.get("agent_id"),

                    "agent_2":
                        r2.get("agent_id"),

                    "task_1":
                        task1,

                    "task_2":
                        task2,

                    "prediction_1":
                        r1.get("prediction"),

                    "prediction_2":
                        r2.get("prediction"),

                    "relation":
                        relation
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

            resolution = resolutions.get(
                task_type,
                {}
            )

            valid_results = [
                r for r in task_results
                if r.get("status") == "success"
            ]

            # -------------------------------------------------------------
            # Decision Engine
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

                    decision = {}

            except Exception:

                decision = {}

            if not isinstance(decision, dict):

                decision = {}

            # DecisionEngine currently uses "decision".
            # Normalize it to "decision_level" for ActionEngine.

            if "decision_level" not in decision:

                decision["decision_level"] = decision.get(
                    "decision",
                    "uncertain"
                )

            # Use task-specific consensus when available.

            task_prediction = resolution.get(
                "prediction"
            )

            if task_prediction is not None:

                decision["prediction"] = (
                    task_prediction
                )

            else:

                decision.setdefault(
                    "prediction",
                    None
                )

            decision.setdefault(
                "status",
                "completed"
            )

            # -------------------------------------------------------------
            # Action Engine
            # -------------------------------------------------------------

            try:

                action = self.action_engine.generate(
                    decision
                )

            except Exception as exc:

                action = {
                    "status": "error",
                    "error": str(exc)
                }

            task_decisions[task_type] = {
                "decision": decision,
                "resolution": resolution,
                "num_agents": len(task_results),
                "num_valid_agents": len(
                    valid_results
                )
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

        successful = [
            r for r in results
            if r.get("status") == "success"
        ]

        failed = [
            r for r in results
            if r.get("status") == "failed"
        ]

        confidences = [
            float(r.get("confidence", 0.0))
            for r in successful
        ]

        trusts = [
            float(r.get("trust", 0.0))
            for r in successful
        ]

        qualities = [
            float(r.get("quality", 0.0))
            for r in successful
        ]

        coverage = (
            len(successful) / len(results)
            if results
            else 0.0
        )

        return {

            "total_agents":
                len(results),

            "successful_agents":
                len(successful),

            "failed_agents":
                len(failed),

            "coverage":
                coverage,

            "mean_confidence":
                (
                    sum(confidences)
                    / len(confidences)
                    if confidences
                    else 0.0
                ),

            "mean_trust":
                (
                    sum(trusts)
                    / len(trusts)
                    if trusts
                    else 0.0
                ),

            "mean_quality":
                (
                    sum(qualities)
                    / len(qualities)
                    if qualities
                    else 0.0
                ),

            "num_conflicts":
                len(conflicts),

            "conflict_rate":
                (
                    len(conflicts)
                    / len(successful)
                    if successful
                    else 0.0
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
    """
    Execute all registered agents with modality-aware input routing.

    Clinical/tabular agents receive their input from `inputs`.
    2D image agents receive their image from `images`.
    3D agents receive their volume from `inputs`.
    """

    # =========================================================
    # NO AGENTS
    # =========================================================

    if not self.agents:

        return {
            "status": "insufficient_evidence",
            "patient_id": patient_id,
            "agents": [],
            "error": "No agents registered."
        }

    inputs = inputs or {}
    images = images or {}

    results = []

    # =========================================================
    # EXECUTE AGENTS
    # =========================================================

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

        # -----------------------------------------------------
        # INPUT ROUTING
        # -----------------------------------------------------

        if isinstance(inputs, dict):

            agent_input = inputs.get(
                agent_id
            )

        else:

            agent_input = inputs

        # -----------------------------------------------------
        # IMAGE ROUTING
        # -----------------------------------------------------

        if isinstance(images, dict):

            agent_image = images.get(
                agent_id
            )

        else:

            agent_image = images

        try:

            # =================================================
            # MODALITY-AWARE PREDICTION
            # =================================================

            if hasattr(agent, "predict"):

                # -------------------------------------------------
                # 2D IMAGE AGENT
                # -------------------------------------------------

                if modality == "2D_image":

                    if agent_image is None:

                        raise ValueError(
                            f"No image provided for image agent "
                            f"'{agent_id}'."
                        )

                    raw_result = agent.predict(
                        agent_image
                    )

                # -------------------------------------------------
                # ALL NON-IMAGE AGENTS
                # -------------------------------------------------

                else:

                    if agent_input is None:

                        # Backward-compatible fallback
                        agent_input = kwargs.get(
                            "patient_data"
                        )

                    if agent_input is None:

                        raise ValueError(
                            f"No input provided for agent "
                            f"'{agent_id}'."
                        )

                    raw_result = agent.predict(
                        agent_input
                    )

            # =================================================
            # ANALYZE-BASED AGENT
            # =================================================

            elif hasattr(agent, "analyze"):

                if agent_input is None:

                    agent_input = kwargs.get(
                        "patient_data"
                    )

                if agent_input is None:

                    raise ValueError(
                        f"No input provided for agent "
                        f"'{agent_id}'."
                    )

                raw_result = agent.analyze(
                    agent_input
                )

            # =================================================
            # INVALID AGENT
            # =================================================

            else:

                raise AttributeError(
                    f"Agent '{agent_id}' has neither "
                    "'predict' nor 'analyze'."
                )

            # =================================================
            # NORMALIZATION
            # =================================================

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

    # =========================================================
    # TRUST
    # =========================================================

    grouped = self._group_by_task(
        results
    )

    for result in results:

        task_results = grouped.get(
            result.get("task_type"),
            []
        )

        if result.get("status") == "success":

            self._compute_trust(
                result,
                task_results
            )

        else:

            result["trust"] = 0.0
            result["agreement"] = 0.0

    # =========================================================
    # ADAPTIVE FUSION
    # =========================================================

    fusion_result = self.fusion.fuse(
        results
    )

    # =========================================================
    # CONFLICT DETECTION
    # =========================================================

    conflicts = self._detect_conflicts(
        results
    )

    # =========================================================
    # CONFLICT RESOLUTION
    # =========================================================

    resolutions = self._resolve_conflicts(
        results,
        conflicts
    )

    # =========================================================
    # EVIDENCE GRAPH
    # =========================================================

    evidence_graph = self._build_evidence_graph(
        results
    )

    # =========================================================
    # TASK DECISIONS + ACTIONS
    # =========================================================

    task_decisions, task_actions = (
        self._build_task_decisions(
            results,
            conflicts,
            resolutions
        )
    )

    # =========================================================
    # REASONING
    # =========================================================

    reasoning_result = {

        "status":
            "completed"
            if results
            else "insufficient_evidence",

        "task_assessments": {

            task: {

                "prediction":
                    resolutions.get(
                        task,
                        {}
                    ).get(
                        "prediction"
                    ),

                "confidence":
                    resolutions.get(
                        task,
                        {}
                    ).get(
                        "consensus_strength",
                        0.0
                    ),

                "resolution":
                    resolutions.get(
                        task,
                        {}
                    )

            }

            for task in grouped
        },

        "evidence_graph":
            evidence_graph,

        "explanation":
            (
                "Evidence was evaluated separately "
                "for each medical task. "
                "Different tasks are complementary "
                "and are not merged into one prediction."
            )
    }

    # =========================================================
    # GLOBAL COORDINATION SUMMARY
    # =========================================================

    coordination = (
        self._build_coordination_summary(
            results,
            conflicts
        )
    )

    # =========================================================
    # DECISION
    # =========================================================

    decision_result = {

        "status":
            "completed"
            if results
            else "insufficient_evidence",

        "task_decisions":
            task_decisions,

        "coordination":
            coordination,

        # IMPORTANT:
        # Heterogeneous medical tasks must not be
        # collapsed into one global class.
        "prediction":
            None,

        "explanation":
            (
                "Decisions are task-specific. "
                "No global medical class is inferred "
                "from heterogeneous agents."
            )
    }

    # =========================================================
    # FINAL RESULT
    # =========================================================

    return {

        "status":
            "completed",

        "patient_id":
            patient_id,

        "agents":
            results,

        "task_summary": {

            task: {

                "num_agents":
                    len(task_results),

                "prediction":
                    resolutions.get(
                        task,
                        {}
                    ).get(
                        "prediction"
                    ),

                "conflict":
                    any(
                        c.get("task_type") == task
                        for c in conflicts
                    )
            }

            for task, task_results
            in grouped.items()
        },

        "conflicts":
            conflicts,

        "conflict_resolution":
            resolutions,

        "fusion":
            fusion_result,

        "reasoning":
            reasoning_result,

        "decision":
            decision_result,

        "action": {
            "task_actions":
                task_actions
        },

        "timestamp":
            datetime.utcnow().isoformat()
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
        # Group ground truth by task
        # ---------------------------------------------------------------------

        if not isinstance(
            ground_truths,
            dict
        ):

            raise ValueError(
                "ground_truths must be a dictionary "
                "indexed by task_type."
            )

        grouped = self._group_by_task(
            agent_results
        )

        feedback_results = {}

        for task_type, task_results in grouped.items():

            truth = ground_truths.get(
                task_type
            )

            if truth is None:
                continue

            try:

                feedback_results[task_type] = (
                    self.feedback_engine.update(
                        agent_results=task_results,
                        ground_truth=truth
                    )
                )

            except Exception as exc:

                feedback_results[task_type] = {
                    "status": "error",
                    "error": str(exc)
                }

        return {
            "status": "completed",
            "tasks": feedback_results
        }

    # =========================================================================
    # BACKWARD COMPATIBILITY FEEDBACK ALIAS
    # =========================================================================

    def feedback(
        self,
        agent_results,
        ground_truth
    ):

        return self.feedback_engine.update(
            agent_results=agent_results,
            ground_truth=ground_truth
        )

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self):

        agent_status = {}

        for agent_id, agent in self.agents.items():

            metadata = self.agent_metadata.get(
                agent_id,
                {}
            )

            agent_status[agent_id] = {

                "registered":
                    True,

                "task_type":
                    metadata.get(
                        "task_type",
                        "unknown"
                    ),

                "modality":
                    metadata.get(
                        "modality",
                        "unknown"
                    ),

                "has_predict":
                    hasattr(
                        agent,
                        "predict"
                    ),

                "has_analyze":
                    hasattr(
                        agent,
                        "analyze"
                    )
            }

        return {

            "status":
                "healthy",

            "num_agents":
                len(self.agents),

            "agents":
                agent_status
        }


# =============================================================================
# BACKWARD-COMPATIBILITY ALIAS
# =============================================================================
#
# Some existing project files may import:
#
#     from coordinator.liver_coordinator import LiverAICoordinator
#
# Keep this alias so those imports continue to work.
#
# =============================================================================

LiverAICoordinator = LiverCoordinator
