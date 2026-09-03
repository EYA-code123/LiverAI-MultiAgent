from coordinator.agent_adapter import AgentAdapter
from coordinator.trust_manager import TrustManager
from coordinator.adaptive_fusion import AdaptiveFusion
from coordinator.conflict_detector import ConflictDetector
from coordinator.conflict_resolver import ConflictResolver
from coordinator.reasoning import EvidenceReasoner
from coordinator.decision import DecisionEngine
from coordinator.action import ActionEngine
from coordinator.feedback import FeedbackEngine


class LiverAICoordinator:

    def __init__(
        self,
        agents=None
    ):

        self.agents = {}

        self.trust_manager = (
            TrustManager()
        )

        self.fusion = (
            AdaptiveFusion()
        )

        self.conflict_detector = (
            ConflictDetector()
        )

        self.conflict_resolver = (
            ConflictResolver()
        )

        self.reasoner = (
            EvidenceReasoner()
        )

        self.decision_engine = (
            DecisionEngine()
        )

        self.action_engine = (
            ActionEngine()
        )

        self.feedback_engine = (
            FeedbackEngine(
                self.trust_manager
            )
        )

        if agents:

            self.register_agents(
                agents
            )

    # =========================================================
    # REGISTER
    # =========================================================

    def register_agent(
        self,
        agent_id,
        agent,
        task_type,
        modality="unknown"
    ):

        adapter = AgentAdapter(

            agent_id=
                agent_id,

            agent=
                agent,

            task_type=
                task_type,

            modality=
                modality
        )

        self.agents[
            agent_id
        ] = adapter

        self.trust_manager.register_agent(
            agent_id
        )

    def register_agents(
        self,
        agents
    ):

        for agent_id, config in (
            agents.items()
        ):

            # -------------------------------------------------
            # Already configured
            # -------------------------------------------------

            if isinstance(
                config,
                AgentAdapter
            ):

                self.agents[
                    agent_id
                ] = config

                continue

            # -------------------------------------------------
            # Dict configuration
            # -------------------------------------------------

            if isinstance(
                config,
                dict
            ):

                self.register_agent(

                    agent_id=
                        agent_id,

                    agent=
                        config.get(
                            "agent"
                        ),

                    task_type=
                        config.get(
                            "task_type",
                            "unknown"
                        ),

                    modality=
                        config.get(
                            "modality",
                            "unknown"
                        )
                )

                continue

            # -------------------------------------------------
            # Raw agent
            # -------------------------------------------------

            self.register_agent(

                agent_id=
                    agent_id,

                agent=
                    config,

                task_type=
                    getattr(
                        config,
                        "task_type",
                        "unknown"
                    ),

                modality=
                    getattr(
                        config,
                        "modality",
                        "unknown"
                    )
            )

    # =========================================================
    # RUN
    # =========================================================

    def run(
        self,
        patient_id,
        inputs=None,
        ground_truth=None
    ):

        inputs = inputs or {}

        # =====================================================
        # PHASE 1
        # =====================================================

        agent_results = []

        for agent_id, adapter in (
            self.agents.items()
        ):

            data = (
                inputs.get(
                    agent_id
                )
            )

            # -------------------------------------------------
            # Missing modality/data
            # -------------------------------------------------

            if data is None:

                result = {

                    "patient_id":
                        patient_id,

                    "agent_id":
                        agent_id,

                    "agent":
                        agent_id,

                    "task_type":
                        adapter.task_type,

                    "modality":
                        adapter.modality,

                    "prediction":
                        None,

                    "probability":
                        None,

                    "class_probabilities":
                        {},

                    "confidence":
                        0.0,

                    "uncertainty":
                        1.0,

                    "quality":
                        0.0,

                    "missing_data_ratio":
                        1.0,

                    "agreement":
                        0.5,

                    "stability":
                        0.5,

                    "utility":
                        0.0,

                    "status":
                        "unavailable",

                    "error":
                        "Required input not provided."
                }

            else:

                result = adapter.predict(

                    patient_id=
                        patient_id,

                    data=
                        data
                )

            # -------------------------------------------------
            # Trust
            # -------------------------------------------------

            if result.get(
                "status"
            ) in (
                "success",
                "completed"
            ):

                result["trust"] = (

                    self.trust_manager
                    .compute_trust(

                        agent_id=
                            agent_id,

                        confidence=
                            result.get(
                                "confidence",
                                0.0
                            ),

                        uncertainty=
                            result.get(
                                "uncertainty",
                                1.0
                            ),

                        quality=
                            result.get(
                                "quality",
                                0.0
                            ),

                        missing_data_ratio=
                            result.get(
                                "missing_data_ratio",
                                1.0
                            ),

                        agreement=
                            result.get(
                                "agreement",
                                0.5
                            ),

                        stability=
                            result.get(
                                "stability",
                                0.5
                            ),

                        utility=
                            result.get(
                                "utility",
                                0.5
                            ),

                        modality_available=
                            True
                    )
                )

            else:

                result["trust"] = 0.0

            agent_results.append(
                result
            )

        # =====================================================
        # PHASE 5
        # =====================================================

        fusion = (
            self.fusion.fuse(
                agent_results
            )
        )

        # =====================================================
        # PHASE 7
        # =====================================================

        conflicts_list = (
            self.conflict_detector.detect(
                agent_results
            )
        )

        # -----------------------------------------------------
        # Group conflicts by task
        # -----------------------------------------------------

        tasks = set(

            r.get(
                "task_type"
            )

            for r in agent_results

            if r.get(
                "prediction"
            ) is not None
        )

        conflict_resolution = {}

        for task in tasks:

            task_results = [

                r

                for r in agent_results

                if r.get(
                    "task_type"
                ) == task
            ]

            task_conflicts = [

                c

                for c in conflicts_list

                if c.get(
                    "task_type"
                ) == task
            ]

            conflict_resolution[
                task
            ] = (

                self.conflict_resolver.resolve(

                    task_type=
                        task,

                    results=
                        task_results,

                    conflicts=
                        task_conflicts
                )
            )

        # =====================================================
        # PHASE 6
        # =====================================================

        reasoning = (
            self.reasoner.synthesize(

                agent_results,

                conflict_resolution=(
                    self._best_resolution(
                        conflict_resolution
                    )
                )
            )
        )

        # =====================================================
        # PHASE 8
        # =====================================================

        decision = (
            self.decision_engine.decide(

                results=
                    agent_results,

                conflicts=
                    conflicts_list,

                reasoning=
                    reasoning
            )
        )

        # =====================================================
        # PHASE 9
        # =====================================================

        action = (
            self.action_engine.generate(
                decision
            )
        )

        # =====================================================
        # PHASE 10
        # =====================================================

        feedback = None

        if ground_truth is not None:

            feedback = (
                self.feedback_engine.update(

                    agent_results=
                        agent_results,

                    ground_truth=
                        ground_truth
                )
            )

        # =====================================================
        # FINAL
        # =====================================================

        return {

            "patient_id":
                patient_id,

            "status":
                "completed",

            "agents":
                agent_results,

            "fusion":
                fusion,

            "conflicts":
                conflicts_list,

            "conflict_resolution":
                conflict_resolution,

            "reasoning":
                reasoning,

            "decision":
                decision,

            "action":
                action,

            "feedback":
                feedback
        }

    # =========================================================
    # BEST RESOLUTION
    # =========================================================

    @staticmethod
    def _best_resolution(
        resolutions
    ):

        for resolution in (
            resolutions.values()
        ):

            if resolution.get(
                "prediction"
            ) is not None:

                return resolution

        return None
