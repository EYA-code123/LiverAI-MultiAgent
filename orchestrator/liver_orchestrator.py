import time

from orchestrator.schemas import AgentResult

from coordinator.trust import (
    AdaptiveTrustManager
)

from coordinator.adaptive_fusion import (
    AdaptiveFusion
)

from coordinator.conflict import (
    ConflictDetector
)

from coordinator.decision import (
    DecisionEngine
)

from coordinator.feedback import (
    FeedbackEngine
)


class LiverAIOrchestrator:

    def __init__(
        self,
        fatty_agent=None,
        fibrosis_agent=None,
        cirrhosis_agent=None,
        tumor_agent=None,
        segmentation_agent=None,
        clinical_reasoning_agent=None
    ):

        self.agents = {

            "fatty_liver":
                fatty_agent,

            "fibrosis":
                fibrosis_agent,

            "cirrhosis":
                cirrhosis_agent,

            "tumor_classification":
                tumor_agent,

            "liver_segmentation":
                segmentation_agent,

            "clinical_reasoning":
                clinical_reasoning_agent
        }

        self.trust_manager = (
            AdaptiveTrustManager()
        )

        self.fusion = (
            AdaptiveFusion()
        )

        self.conflict_detector = (
            ConflictDetector()
        )

        self.decision_engine = (
            DecisionEngine()
        )

        self.feedback_engine = (
            FeedbackEngine(
                self.trust_manager
            )
        )

        self.last_results = []
        self.last_trust = {}
        self.last_conflicts = []
        self.last_fusion = None
        self.last_decision = None

    def _execute_agent(
        self,
        agent_id,
        agent,
        data
    ):

        if agent is None:

            return AgentResult(
                agent_id=agent_id,
                error="Agent unavailable"
            )

        if data is None:

            return AgentResult(
                agent_id=agent_id,
                error="Input unavailable"
            )

        start = time.perf_counter()

        try:

            if hasattr(
                agent,
                "predict"
            ):

                raw = agent.predict(
                    data
                )

            elif hasattr(
                agent,
                "analyze"
            ):

                raw = agent.analyze(
                    data
                )

            else:

                raise AttributeError(
                    "Agent must implement "
                    "predict() or analyze()"
                )

            latency = (
                time.perf_counter()
                - start
            ) * 1000.0

            if not isinstance(
                raw,
                dict
            ):

                raw = {
                    "prediction": raw
                }

            return AgentResult(

                agent_id=agent_id,

                task_type=raw.get(
                    "task_type",
                    agent_id
                ),

                prediction=raw.get(
                    "prediction"
                ),

                probability=raw.get(
                    "probability"
                ),

                confidence=float(
                    raw.get(
                        "confidence",
                        0.5
                    )
                ),

                uncertainty=float(
                    raw.get(
                        "uncertainty",
                        0.5
                    )
                ),

                quality=float(
                    raw.get(
                        "quality",
                        1.0
                    )
                ),

                missing_data_ratio=float(
                    raw.get(
                        "missing_data_ratio",
                        0.0
                    )
                ),

                latency_ms=latency,

                explanation=raw.get(
                    "explanation"
                ),

                details=raw.get(
                    "details",
                    {}
                ),

                error=raw.get(
                    "error"
                )
            )

        except Exception as e:

            return AgentResult(
                agent_id=agent_id,
                error=str(e)
            )

    def assess_agents(
        self,
        results
    ):

        trusts = {}

        for result in results:

            if result.status != "success":
                continue

            if result.agent_id not in (
                self.trust_manager
                .historical_performance
            ):

                self.trust_manager.register_agent(
                    result.agent_id,
                    performance=0.5
                )

            trust = (
                self.trust_manager
                .compute_trust(

                    result.agent_id,

                    confidence=
                        result.confidence,

                    uncertainty=
                        result.uncertainty,

                    quality=
                        result.quality,

                    agreement=1.0,

                    modality_availability=(
                        1.0
                        - result.missing_data_ratio
                    )
                )
            )

            result.trust = trust

            trusts[
                result.agent_id
            ] = trust

        return trusts

    def predict(
        self,
        clinical_data=None,
        tumor_image=None,
        liver_volume=None
    ):

        results = []

        inputs = {

            "fatty_liver":
                clinical_data,

            "fibrosis":
                clinical_data,

            "cirrhosis":
                clinical_data,

            "tumor_classification":
                tumor_image,

            "liver_segmentation":
                liver_volume,

            "clinical_reasoning":
                clinical_data
        }

        for agent_id, agent in (
            self.agents.items()
        ):

            result = self._execute_agent(

                agent_id,

                agent,

                inputs.get(
                    agent_id
                )
            )

            results.append(
                result
            )

        # ----------------------------------
        # 1. ASSESSMENT + TRUST
        # ----------------------------------

        trusts = self.assess_agents(
            results
        )

        # ----------------------------------
        # 2. CONFLICT DETECTION
        # ----------------------------------

        conflicts = (
            self.conflict_detector
            .detect(results)
        )

        consensus = (
            self.conflict_detector
            .consensus(results)
        )

        # ----------------------------------
        # 3. ADAPTIVE FUSION
        # ----------------------------------

        fusion = (
            self.fusion
            .fuse(results)
        )

        # ----------------------------------
        # 4. DECISION
        # ----------------------------------

        decision = (
            self.decision_engine
            .decide(

                results,

                conflicts,

                fusion,

                consensus
            )
        )

        self.last_results = results
        self.last_trust = trusts
        self.last_conflicts = conflicts
        self.last_fusion = fusion
        self.last_decision = decision

        return {

            "agents": [
                r.to_dict()
                for r in results
            ],

            "trust_scores":
                trusts,

            "conflicts":
                conflicts,

            "consensus":
                consensus,

            "adaptive_fusion":
                fusion,

            "decision":
                decision
        }

    def update_feedback(
        self,
        ground_truth
    ):

        return (
            self.feedback_engine
            .update(
                self.last_results,
                ground_truth
            )
        )
