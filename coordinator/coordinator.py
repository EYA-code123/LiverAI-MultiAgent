from .trust import TrustManager
from .conflict import ConflictDetector
from .fusion import FusionEngine
from .decision import DecisionEngine


class LiverCoordinator:

    def __init__(self):

        self.trust_manager = TrustManager()

        self.conflict_detector = (
            ConflictDetector()
        )

        self.fusion_engine = (
            FusionEngine()
        )

        self.decision_engine = (
            DecisionEngine()
        )

    # ---------------------------------------------------------
    # REGISTER AGENT PERFORMANCE
    # ---------------------------------------------------------

    def register_agent_performance(
        self,
        agent_id,
        performance
    ):

        self.trust_manager.register_agent(
            agent_id,
            performance
        )

    # ---------------------------------------------------------
    # COMPUTE TRUST
    # ---------------------------------------------------------

    def compute_trust_scores(
        self,
        messages
    ):

        trust_scores = {}

        for message in messages:

            trust = (
                self.trust_manager.compute_trust(
                    agent_id=message.agent_id,
                    confidence=message.confidence,
                    quality=message.quality,
                    uncertainty=message.uncertainty
                )
            )

            trust_scores[
                message.agent_id
            ] = trust

        return trust_scores

    # ---------------------------------------------------------
    # COORDINATE
    # ---------------------------------------------------------

    def coordinate(self, messages):

        trust_scores = (
            self.compute_trust_scores(
                messages
            )
        )

        conflicts = (
            self.conflict_detector.detect(
                messages
            )
        )

        # Groupement par type de tâche
        task_groups = {}

        for message in messages:

            task_type = message.details.get(
                "task_type",
                "unknown"
            )

            task_groups.setdefault(
                task_type,
                []
            ).append(message)

        fused_results = {}

        for task_type, group in task_groups.items():

            if task_type == "classification":

                fused_results[task_type] = (
                    self.fusion_engine.fuse_classification(
                        group,
                        trust_scores
                    )
                )

        decision = (
            self.decision_engine.decide(
                agent_results=messages,
                conflicts=conflicts,
                fused_results=fused_results
            )
        )

        return {
            "trust_scores": trust_scores,
            "conflicts": conflicts,
            "fused_results": fused_results,
            "decision": decision
        } 
