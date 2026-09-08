# =============================================================================
# LIVER AI — LIVER COORDINATOR
# =============================================================================
#
# Task-aware multi-agent coordinator for heterogeneous liver AI agents.
#
# Pipeline:
#
#   Agents
#      ↓
#   Normalization
#      ↓
#   Trust
#      ↓
#   Conflict Detection
#      ↓
#   Conflict Resolution
#      ↓
#   Adaptive Fusion
#      ↓
#   Evidence Reasoning
#      ↓
#   Decision Engine
#      ↓
#   Action Engine
#      ↓
#   Final Multi-Agent Result
#
# IMPORTANT:
# Different medical tasks are NOT merged into one global prediction.
#
# =============================================================================
from coordinator.communication import (
    CommunicationProtocol,
    AgentMessage,
)
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
        self.feedback_engine = FeedbackEngine()

        self.communication = CommunicationProtocol()

        self.coordination_trace = []

        self.delegation_history = []
        # ---------------------------------------------------------------------
        # Optional initial agents
        # ---------------------------------------------------------------------

        if agents:
            self._load_agents(agents)

    # =========================================================================
    # LOAD AGENTS
    # =========================================================================

    def _load_agents(self, agents):

        if not isinstance(agents, dict):
            return

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

        # ---------------------------------------------------------------------
        # Register initial trust
        # ---------------------------------------------------------------------

        try:

            self.trust_manager.register_agent(
                agent_id=agent_id,
                performance=0.5
            )

        except Exception:

            # Agent may already exist in TrustManager.
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

        return list(
            self.agents.keys()
        )

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

        # ---------------------------------------------------------------------
        # None result
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # Non-dictionary result
        # ---------------------------------------------------------------------

        if not isinstance(result, dict):

            result = {

                "prediction": result
            }

        # ---------------------------------------------------------------------
        # Basic values
        # ---------------------------------------------------------------------

        prediction = result.get(
            "prediction"
        )

        probability = result.get(
            "probability"
        )

        confidence = result.get(
            "confidence"
        )

        # ---------------------------------------------------------------------
        # Confidence fallback
        # ---------------------------------------------------------------------

        if confidence is None:

            if probability is not None:

                try:

                    if isinstance(
                        probability,
                        (list, tuple)
                    ):

                        confidence = max(
                            probability
                        )

                    else:

                        confidence = float(
                            probability
                        )

                except Exception:

                    confidence = 0.0

            else:

                confidence = 0.0

        # ---------------------------------------------------------------------
        # Safe float conversion
        # ---------------------------------------------------------------------

        try:

            confidence = float(
                confidence
            )

        except Exception:

            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence
            )
        )

        # ---------------------------------------------------------------------
        # Uncertainty
        # ---------------------------------------------------------------------

        uncertainty = result.get(
            "uncertainty"
        )

        if uncertainty is None:

            uncertainty = 1.0 - confidence

        try:

            uncertainty = float(
                uncertainty
            )

        except Exception:

            uncertainty = 1.0 - confidence

        uncertainty = max(
            0.0,
            min(
                1.0,
                uncertainty
            )
        )

        # ---------------------------------------------------------------------
        # Quality
        # ---------------------------------------------------------------------

        quality = result.get(
            "quality",
            1.0
        )

        try:

            quality = float(
                quality
            )

        except Exception:

            quality = 1.0

        quality = max(
            0.0,
            min(
                1.0,
                quality
            )
        )

        # ---------------------------------------------------------------------
        # Missing data
        # ---------------------------------------------------------------------

        missing_ratio = result.get(
            "missing_data_ratio",
            result.get(
                "missing_ratio",
                0.0
            )
        )

        try:

            missing_ratio = float(
                missing_ratio
            )

        except Exception:

            missing_ratio = 0.0

        missing_ratio = max(
            0.0,
            min(
                1.0,
                missing_ratio
            )
        )

        # ---------------------------------------------------------------------
        # Status
        # ---------------------------------------------------------------------

        status = result.get(
            "status",
            "success"
        )

        if status not in [
            "success",
            "failed",
            "error",
            "unavailable"
        ]:

            status = "success"

        # ---------------------------------------------------------------------
        # Normalized result
        # ---------------------------------------------------------------------

        return {

            "agent_id":
                agent_id,

            "agent":
                result.get(
                    "agent",
                    agent_id
                ),

            "task_type":
                result.get(
                    "task_type",
                    task_type
                ),

            "modality":
                result.get(
                    "modality",
                    modality
                ),

            "prediction":
                (
                    str(prediction)
                    if prediction is not None
                    else None
                ),

            "probability":
                probability,

            "confidence":
                confidence,

            "uncertainty":
                uncertainty,

            "quality":
                quality,

            "missing_data_ratio":
                missing_ratio,

            "status":
                status,

            "error":
                result.get(
                    "error"
                ),

            "details":
                result
        }

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

        return dict(
            grouped
        )

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

        # ---------------------------------------------------------------------
        # Trust Manager
        # ---------------------------------------------------------------------

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
                    +
                    0.3 * quality
                    +
                    0.2 * (
                        1.0 - uncertainty
                    )
                )

        except Exception:

            trust = (
                0.5 * confidence
                +
                0.3 * quality
                +
                0.2 * (
                    1.0 - uncertainty
                )
            )

        # ---------------------------------------------------------------------
        # Clamp trust
        # ---------------------------------------------------------------------

        try:

            trust = float(
                trust
            )

        except Exception:

            trust = 0.0

        result["trust"] = max(
            0.0,
            min(
                1.0,
                trust
            )
        )

        # ---------------------------------------------------------------------
        # Agreement
        #
        # Agreement is calculated ONLY against agents performing
        # the SAME medical task.
        # ---------------------------------------------------------------------

        same_task = [

            r

            for r in task_results

            if r.get(
                "agent_id"
            ) != result.get(
                "agent_id"
            )

            and r.get(
                "status"
            ) == "success"

            and r.get(
                "prediction"
            ) is not None
        ]

        if not same_task:

            result["agreement"] = 0.5

        else:

            agreements = [

                1.0

                if r.get(
                    "prediction"
                )
                ==
                result.get(
                    "prediction"
                )

                else 0.0

                for r in same_task
            ]

            result["agreement"] = (

                sum(
                    agreements
                )
                /
                len(
                    agreements
                )
            )

    # =========================================================================
    # DETECT CONFLICTS
    # =========================================================================

    def _detect_conflicts(self, results):

        try:

            detected = (
                self.conflict_detector.detect(
                    results
                )
            )

            if detected is None:

                return []

            return detected

        except Exception:

            conflicts = []

            grouped = self._group_by_task(
                results
            )

            for task_type, task_results in grouped.items():

                valid = [

                    r

                    for r in task_results

                    if r.get(
                        "status"
                    ) == "success"

                    and r.get(
                        "prediction"
                    ) is not None
                ]

                for i in range(
                    len(valid)
                ):

                    for j in range(
                        i + 1,
                        len(valid)
                    ):

                        r1 = valid[i]

                        r2 = valid[j]

                        if (

                            r1.get(
                                "prediction"
                            )
                            !=
                            r2.get(
                                "prediction"
                            )
                        ):

                            conflicts.append({

                                "task_type":
                                    task_type,

                                "agent_1":
                                    r1.get(
                                        "agent_id"
                                    ),

                                "prediction_1":
                                    r1.get(
                                        "prediction"
                                    ),

                                "confidence_1":
                                    r1.get(
                                        "confidence",
                                        0.0
                                    ),

                                "agent_2":
                                    r2.get(
                                        "agent_id"
                                    ),

                                "prediction_2":
                                    r2.get(
                                        "prediction"
                                    ),

                                "confidence_2":
                                    r2.get(
                                        "confidence",
                                        0.0
                                    )
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

                c

                for c in conflicts

                if c.get(
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

                    "status":
                        "error",

                    "consensus":
                        False,

                    "prediction":
                        None,

                    "consensus_strength":
                        0.0,

                    "scores":
                        {},

                    "reason":
                        str(exc)
                }

            if not isinstance(
                resolution,
                dict
            ):

                resolution = {

                    "status":
                        "error",

                    "consensus":
                        False,

                    "prediction":
                        None,

                    "consensus_strength":
                        0.0,

                    "scores":
                        {},

                    "reason":
                        "Invalid conflict resolution result."
                }

            resolutions[task_type] = (
                resolution
            )

        return resolutions

    # =========================================================================
    # TASK-AWARE EVIDENCE GRAPH
    # =========================================================================

    def _build_evidence_graph(self, results):

        graph = []

        valid_results = [

            r

            for r in results

            if r.get(
                "status"
            ) == "success"
        ]

        for i in range(
            len(valid_results)
        ):

            for j in range(
                i + 1,
                len(valid_results)
            ):

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
                # Same task + same prediction = support
                # -------------------------------------------------------------

                elif (
                    r1.get(
                        "prediction"
                    )
                    ==
                    r2.get(
                        "prediction"
                    )
                ):

                    relation = "supports"

                # -------------------------------------------------------------
                # Same task + different prediction = conflict
                # -------------------------------------------------------------

                else:

                    relation = "conflicts"

                graph.append({

                    "agent_1":
                        r1.get(
                            "agent_id"
                        ),

                    "agent_2":
                        r2.get(
                            "agent_id"
                        ),

                    "task_1":
                        task1,

                    "task_2":
                        task2,

                    "prediction_1":
                        r1.get(
                            "prediction"
                        ),

                    "prediction_2":
                        r2.get(
                            "prediction"
                        ),

                    "relation":
                        relation
                })

        return graph

    # =========================================================================
    # TASK DECISIONS + ACTIONS
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

        # =====================================================================
        # 1. BUILD ALL TASK DECISIONS
        # =====================================================================

        for task_type, task_results in grouped.items():

            resolution = resolutions.get(
                task_type,
                {}
            )

            valid_results = [

                r

                for r in task_results

                if r.get(
                    "status"
                ) == "success"
            ]

            # -----------------------------------------------------------------
            # Decision Engine
            # -----------------------------------------------------------------

            try:

                decision = (
                    self.decision_engine.decide(
                        task_results
                    )
                )

            except TypeError:

                try:

                    decision = (
                        self.decision_engine.decide(
                            results=task_results
                        )
                    )

                except Exception:

                    decision = {}

            except Exception:

                decision = {}

            if not isinstance(
                decision,
                dict
            ):

                decision = {}

            # -----------------------------------------------------------------
            # Decision level
            # -----------------------------------------------------------------

            decision_level = decision.get(

                "decision_level",

                decision.get(
                    "decision",
                    "UNCERTAIN"
                )
            )

            if isinstance(
                decision_level,
                str
            ):

                decision_level = (
                    decision_level.upper()
                )

            else:

                decision_level = "UNCERTAIN"

            decision[
                "decision_level"
            ] = decision_level

            # -----------------------------------------------------------------
            # Consensus prediction
            # -----------------------------------------------------------------

            task_prediction = resolution.get(
                "prediction"
            )

            if task_prediction is not None:

                decision[
                    "prediction"
                ] = task_prediction

            else:

                decision.setdefault(
                    "prediction",
                    None
                )

            # -----------------------------------------------------------------
            # Confidence
            # -----------------------------------------------------------------

            if decision.get(
                "confidence"
            ) is None:

                decision[
                    "confidence"
                ] = resolution.get(
                    "consensus_strength",
                    0.0
                )

            try:

                decision[
                    "confidence"
                ] = float(
                    decision.get(
                        "confidence",
                        0.0
                    ) or 0.0
                )

            except Exception:

                decision[
                    "confidence"
                ] = 0.0

            decision[
                "confidence"
            ] = max(
                0.0,
                min(
                    1.0,
                    decision[
                        "confidence"
                    ]
                )
            )

            # -----------------------------------------------------------------
            # Risk score
            # -----------------------------------------------------------------

            if decision.get(
                "risk_score"
            ) is None:

                decision[
                    "risk_score"
                ] = 0.0

            try:

                decision[
                    "risk_score"
                ] = float(
                    decision.get(
                        "risk_score",
                        0.0
                    ) or 0.0
                )

            except Exception:

                decision[
                    "risk_score"
                ] = 0.0

            decision[
                "risk_score"
            ] = max(
                0.0,
                min(
                    1.0,
                    decision[
                        "risk_score"
                    ]
                )
            )

            # -----------------------------------------------------------------
            # Additional tests
            # -----------------------------------------------------------------

            decision.setdefault(
                "request_additional_tests",
                False
            )

            decision[
                "request_additional_tests"
            ] = bool(
                decision[
                    "request_additional_tests"
                ]
            )

            # -----------------------------------------------------------------
            # Status
            # -----------------------------------------------------------------

            decision.setdefault(
                "status",
                "completed"
            )

            # =================================================================
            # STORE TASK DECISION
            # =================================================================

            task_decisions[task_type] = {

                "decision":
                    decision,

                "decision_level":
                    decision.get(
                        "decision_level",
                        "UNCERTAIN"
                    ),

                "prediction":
                    decision.get(
                        "prediction"
                    ),

                "confidence":
                    decision.get(
                        "confidence",
                        0.0
                    ),

                "risk_score":
                    decision.get(
                        "risk_score",
                        0.0
                    ),

                "request_additional_tests":
                    decision.get(
                        "request_additional_tests",
                        False
                    ),

                "resolution":
                    resolution,

                "num_agents":
                    len(
                        task_results
                    ),

                "num_valid_agents":
                    len(
                        valid_results
                    )
            }

        # =====================================================================
        # 2. SEND ALL TASK DECISIONS TO ACTION ENGINE
        # =====================================================================

        action_input = {

            "task_decisions": {}
        }

        for task_type, task_data in task_decisions.items():

            action_input[
                "task_decisions"
            ][task_type] = {

                "decision_level":
                    task_data.get(
                        "decision_level",
                        "UNCERTAIN"
                    ),

                "prediction":
                    task_data.get(
                        "prediction"
                    ),

                "confidence":
                    task_data.get(
                        "confidence",
                        0.0
                    ),

                "risk_score":
                    task_data.get(
                        "risk_score",
                        0.0
                    ),

                "request_additional_tests":
                    task_data.get(
                        "request_additional_tests",
                        False
                    )
            }

        # =====================================================================
        # 3. ACTION ENGINE
        # =====================================================================

        try:

            action_result = (
                self.action_engine.generate(
                    action_input
                )
            )

        except Exception as exc:

            action_result = {

                "status":
                    "error",

                "error":
                    str(exc),

                "task_actions":
                    {}
            }

        if not isinstance(
            action_result,
            dict
        ):

            action_result = {

                "status":
                    "error",

                "error":
                    "Invalid ActionEngine output.",

                "task_actions":
                    {}
            }

        task_actions = action_result.get(
            "task_actions",
            {}
        )

        if not isinstance(
            task_actions,
            dict
        ):

            task_actions = {}

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

            r

            for r in results

            if r.get(
                "status"
            ) == "success"
        ]

        failed = [

            r

            for r in results

            if r.get(
                "status"
            ) in [
                "failed",
                "error"
            ]
        ]

        confidences = [

            float(
                r.get(
                    "confidence",
                    0.0
                )
            )

            for r in successful
        ]

        trusts = [

            float(
                r.get(
                    "trust",
                    0.0
                )
            )

            for r in successful
        ]

        qualities = [

            float(
                r.get(
                    "quality",
                    0.0
                )
            )

            for r in successful
        ]

        coverage = (

            len(
                successful
            )
            /
            len(
                results
            )

            if results

            else 0.0
        )

        return {

            "total_agents":
                len(
                    results
                ),

            "successful_agents":
                len(
                    successful
                ),

            "failed_agents":
                len(
                    failed
                ),

            "coverage":
                coverage,

            "mean_confidence":
                (
                    sum(
                        confidences
                    )
                    /
                    len(
                        confidences
                    )
                    if confidences
                    else 0.0
                ),

            "mean_trust":
                (
                    sum(
                        trusts
                    )
                    /
                    len(
                        trusts
                    )
                    if trusts
                    else 0.0
                ),

            "mean_quality":
                (
                    sum(
                        qualities
                    )
                    /
                    len(
                        qualities
                    )
                    if qualities
                    else 0.0
                ),

            "num_conflicts":
                len(
                    conflicts
                ),

            "conflict_rate":
                (
                    len(
                        conflicts
                    )
                    /
                    len(
                        successful
                    )
                    if successful
                    else 0.0
                )
        }
    # ============================================================
# COMMUNICATION PROTOCOL
# ============================================================

def _send_request(
    self,
    agent_id,
    request_type,
    task_type=None,
    payload=None,
    patient_id=None,
):
    """
    Send a structured request from Coordinator to an agent.
    """

    message = self.communication.create_request(
        receiver=agent_id,
        request_type=request_type,
        task_type=task_type,
        payload=payload,
        patient_id=patient_id,
    )

    self.coordination_trace.append(
        message.to_dict()
    )

    return message


def _record_agent_response(
    self,
    agent_id,
    request_message,
    result,
    task_type=None,
    patient_id=None,
):
    """
    Record an agent response in the communication protocol.
    """

    message = self.communication.record_agent_response(
        sender=agent_id,
        request_message=request_message,
        payload=result,
        task_type=task_type,
        patient_id=patient_id,
    )

    self.coordination_trace.append(
        message.to_dict()
    )

    return message


# ============================================================
# DYNAMIC DELEGATION
# ============================================================

def _delegate_task(
    self,
    agent_id,
    request_type,
    agent_input,
    task_type=None,
    patient_id=None,
):
    """
    Dynamically delegate a task to a specialist agent.
    """

    if agent_id not in self.agents:

        return {
            "status": "error",
            "error": f"Agent not registered: {agent_id}",
        }

    request_message = self._send_request(
        agent_id=agent_id,
        request_type=request_type,
        task_type=task_type,
        payload={
            "reason": "dynamic_coordination",
            "request": request_type,
        },
        patient_id=patient_id,
    )

    agent = self.agents[agent_id]

    try:

        raw_result = agent.predict(agent_input)

        normalized_result = self._normalize_result(
            agent_id,
            raw_result,
        )

        self._record_agent_response(
            agent_id=agent_id,
            request_message=request_message,
            result=normalized_result,
            task_type=task_type,
            patient_id=patient_id,
        )

        self.delegation_history.append(
            {
                "request_type": request_type,
                "agent_id": agent_id,
                "task_type": task_type,
                "status": "success",
            }
        )

        return normalized_result

    except Exception as exc:

        error_result = {
            "agent_id": agent_id,
            "agent": agent_id,
            "task_type": task_type,
            "status": "failed",
            "error": str(exc),
        }

        self._record_agent_response(
            agent_id=agent_id,
            request_message=request_message,
            result=error_result,
            task_type=task_type,
            patient_id=patient_id,
        )

        self.delegation_history.append(
            {
                "request_type": request_type,
                "agent_id": agent_id,
                "task_type": task_type,
                "status": "failed",
                "error": str(exc),
            }
        )

        return error_result
        # ============================================================
# COORDINATION POLICY
# ============================================================

def _build_coordination_plan(
    self,
    results,
    inputs=None,
    images=None,
    patient_id=None,
):
    """
    Build a dynamic coordination plan from previous
    agent evidence.

    The Coordinator does not blindly execute every agent.
    It decides whether additional evidence is required.
    """

    plan = []

    results_by_task = {
        r.get("task_type"): r
        for r in results
        if r.get("status") == "success"
    }

    # --------------------------------------------------------
    # 1. TUMOR -> SEGMENTATION
    # --------------------------------------------------------

    tumor_result = results_by_task.get(
        "tumor_classification"
    )

    segmentation_result = results_by_task.get(
        "liver_segmentation"
    )

    if tumor_result is not None:

        prediction = str(
            tumor_result.get("prediction", "")
        ).lower()

        confidence = float(
            tumor_result.get("confidence", 0.0) or 0.0
        )

        suspicious = (
            "carcinoma" in prediction
            or "tumor" in prediction
            or "cancer" in prediction
            or "angiosarcoma" in prediction
            or "cholangiocarcinoma" in prediction
            or "hemangioma" in prediction
        )

        if suspicious and segmentation_result is None:

            plan.append(
                {
                    "request_type":
                        AgentMessage.REQUEST_SEGMENTATION,

                    "agent_id":
                        "LiverSegmentationAgent",

                    "task_type":
                        "liver_segmentation",

                    "reason":
                        "Tumor agent produced a suspicious imaging finding.",

                    "trigger_confidence":
                        confidence,
                }
            )

    # --------------------------------------------------------
    # 2. LOW CONFIDENCE -> REASSESSMENT
    # --------------------------------------------------------

    for result in results:

        if result.get("status") != "success":
            continue

        confidence = float(
            result.get("confidence", 0.0) or 0.0
        )

        uncertainty = float(
            result.get("uncertainty", 1.0) or 1.0
        )

        agent_id = result.get("agent_id")

        task_type = result.get("task_type")

        if confidence < 0.60 or uncertainty > 0.40:

            plan.append(
                {
                    "request_type":
                        AgentMessage.REQUEST_REASSESSMENT,

                    "agent_id":
                        agent_id,

                    "task_type":
                        task_type,

                    "reason":
                        "Low confidence or high uncertainty.",

                    "confidence":
                        confidence,

                    "uncertainty":
                        uncertainty,
                }
            )

    # --------------------------------------------------------
    # 3. HIGH RISK TUMOR -> CLINICAL SUPPORT
    # --------------------------------------------------------

    if tumor_result is not None:

        prediction = str(
            tumor_result.get("prediction", "")
        ).lower()

        if (
            "carcinoma" in prediction
            or "cancer" in prediction
        ):

            clinical_result = results_by_task.get(
                "clinical_reasoning"
            )

            if clinical_result is None:

                plan.append(
                    {
                        "request_type":
                            AgentMessage.REQUEST_ADDITIONAL_EVIDENCE,

                        "agent_id":
                            "ClinicalReasoningAgent",

                        "task_type":
                            "clinical_reasoning",

                        "reason":
                            "Tumor finding requires additional clinical evidence.",
                    }
                )

    return plan
    # ============================================================
# EXECUTE COORDINATION PLAN
# ============================================================

def _execute_coordination_plan(
    self,
    plan,
    inputs=None,
    images=None,
    patient_id=None,
):
    """
    Execute dynamically generated delegation requests.
    """

    additional_results = []

    inputs = inputs or {}
    images = images or {}

    for request in plan:

        agent_id = request["agent_id"]

        request_type = request["request_type"]

        task_type = request.get("task_type")

        # ----------------------------------------------------
        # BUILD INPUT
        # ----------------------------------------------------

        if agent_id in images:

            agent_input = images[agent_id]

        else:

            agent_input = inputs.get(agent_id)

        # ----------------------------------------------------
        # SPECIAL SEGMENTATION ROUTING
        # ----------------------------------------------------

        if (
            task_type == "liver_segmentation"
            and isinstance(agent_input, dict)
            and "image" in agent_input
        ):

            agent_input = agent_input["image"]

        # ----------------------------------------------------
        # MISSING INPUT
        # ----------------------------------------------------

        if agent_input is None:

            additional_results.append(
                {
                    "agent_id": agent_id,
                    "task_type": task_type,
                    "status": "not_run",
                    "error": "No input available for delegated request.",
                    "request_type": request_type,
                }
            )

            continue

        # ----------------------------------------------------
        # EXECUTE
        # ----------------------------------------------------

        result = self._delegate_task(
            agent_id=agent_id,
            request_type=request_type,
            agent_input=agent_input,
            task_type=task_type,
            patient_id=patient_id,
        )

        if isinstance(result, dict):

            result["coordination_request"] = request_type

            result["coordination_reason"] = request.get(
                "reason"
            )

        additional_results.append(result)

    return additional_results
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

        Clinical/tabular agents:
            inputs[agent_id]

        2D image agents:
            images[agent_id]

        Other agents:
            inputs[agent_id]

        The pipeline preserves task-specific outputs and does not
        collapse heterogeneous medical tasks into one global class.
        """

        # =====================================================================
        # NO AGENTS
        # =====================================================================

        if not self.agents:

            return {

                "status":
                    "insufficient_evidence",

                "patient_id":
                    patient_id,

                "agents":
                    [],

                "error":
                    "No agents registered."
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
            # INPUT ROUTING
            # -----------------------------------------------------------------

            if isinstance(
                inputs,
                dict
            ):

                agent_input = inputs.get(
                    agent_id
                )

            else:

                agent_input = inputs

            # -----------------------------------------------------------------
            # IMAGE ROUTING
            # -----------------------------------------------------------------

            if isinstance(
                images,
                dict
            ):

                agent_image = images.get(
                    agent_id
                )

            else:

                agent_image = images

            try:

                # =============================================================
                # PREDICT
                # =============================================================

                if hasattr(
                    agent,
                    "predict"
                ):

                    # ---------------------------------------------------------
                    # 2D IMAGE
                    # ---------------------------------------------------------

                    if modality == "2D_image":

                        if agent_image is None:

                            raise ValueError(

                                f"No image provided for image agent "
                                f"'{agent_id}'."
                            )

                        raw_result = agent.predict(
                            agent_image
                        )

                    # ---------------------------------------------------------
                    # 3D IMAGE
                    # ---------------------------------------------------------

                    elif modality == "3D_CT":

                        volume = (
                            agent_input
                            if agent_input is not None
                            else agent_image
                        )

                        if volume is None:

                            raise ValueError(

                                f"No 3D volume provided for agent "
                                f"'{agent_id}'."
                            )

                        raw_result = agent.predict(
                            volume
                        )

                    # ---------------------------------------------------------
                    # OTHER AGENTS
                    # ---------------------------------------------------------

                    else:

                        if agent_input is None:

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

                # =============================================================
                # ANALYZE
                # =============================================================

                elif hasattr(
                    agent,
                    "analyze"
                ):

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

                # =============================================================
                # INVALID AGENT
                # =============================================================

                else:

                    raise AttributeError(

                        f"Agent '{agent_id}' has neither "
                        "'predict' nor 'analyze'."
                    )

                # =============================================================
                # NORMALIZATION
                # =============================================================

                normalized = self._normalize_result(

                    agent_id=agent_id,

                    result=raw_result,

                    task_type=task_type,

                    modality=modality
                )

            except Exception as exc:

                normalized = {

                    "agent_id":
                        agent_id,

                    "agent":
                        agent_id,

                    "task_type":
                        task_type,

                    "modality":
                        modality,

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

                    "status":
                        "failed",

                    "error":
                        str(exc),

                    "details":
                        {}
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

                "status":
                    "error",

                "error":
                    str(exc)
            }

        # =====================================================================
        # CONFLICT DETECTION
        # =====================================================================

        conflicts = self._detect_conflicts(
            results
        )

        # =====================================================================
        # CONFLICT RESOLUTION
        # =====================================================================

        resolutions = self._resolve_conflicts(

            results,

            conflicts
        )

        # =====================================================================
        # EVIDENCE GRAPH
        # =====================================================================

        evidence_graph = (
            self._build_evidence_graph(
                results
            )
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

            "status":
                (
                    "completed"
                    if results
                    else
                    "insufficient_evidence"
                ),

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

        # =====================================================================
        # COORDINATION SUMMARY
        # =====================================================================

        coordination = (
            self._build_coordination_summary(

                results,

                conflicts
            )
        )

        # =====================================================================
        # DECISION RESULT
        # =====================================================================

        decision_result = {

            "status":
                (
                    "completed"
                    if results
                    else
                    "insufficient_evidence"
                ),

            "task_decisions":
                task_decisions,

            "coordination":
                coordination,

            "prediction":
                None,

            "explanation":
                (
                    "Decisions are task-specific. "
                    "No global medical class is inferred "
                    "from heterogeneous agents."
                )
        }

        # =====================================================================
        # FINAL RESULT
        # =====================================================================

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
                        len(
                            task_results
                        ),

                    "prediction":
                        resolutions.get(
                            task,
                            {}
                        ).get(
                            "prediction"
                        ),

                    "conflict":
                        any(

                            c.get(
                                "task_type"
                            )
                            ==
                            task

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

                "status":
                    (
                        "completed"
                        if task_actions
                        else
                        "incomplete"
                    ),

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

                "status":
                    "no_results"
            }

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

                    "status":
                        "error",

                    "error":
                        str(exc)
                }

        return {

            "status":
                "completed",

            "tasks":
                feedback_results
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
                len(
                    self.agents
                ),

            "agents":
                agent_status
        }


# =============================================================================
# BACKWARD COMPATIBILITY ALIAS
# =============================================================================

LiverAICoordinator = LiverCoordinator
