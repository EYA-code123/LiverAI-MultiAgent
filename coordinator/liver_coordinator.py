from coordinator.trust_manager import TrustManager
from coordinator.conflict_detector import ConflictDetector
from coordinator.adaptive_fusion import AdaptiveFusion


class LiverCoordinator:

    def __init__(self, agents):

        self.agents = agents

        self.trust_manager = TrustManager()

        self.conflict_detector = (
            ConflictDetector()
        )

        self.fusion = AdaptiveFusion()

    def run(self, patient_data):

        results = []

        # ------------------------------------------------
        # 1. ASK ALL RELEVANT AGENTS
        # ------------------------------------------------

        for agent in self.agents:

            try:

                result = agent.predict(
                    patient_data
                )

                results.append(result)

            except Exception as e:

                from orchestrator.schemas import AgentResult

                results.append(
                    AgentResult(
                        agent=agent.name,
                        status="error",
                        error=str(e)
                    )
                )

        # ------------------------------------------------
        # 2. COMPUTE TRUST
        # ------------------------------------------------

        for result in results:

            if result.status != "success":
                continue

            result.trust = (
                self.trust_manager.compute_trust(
                    agent_name=result.agent,
                    confidence=result.confidence or 0.0,
                    quality=result.quality or 1.0
                )
            )

        # ------------------------------------------------
        # 3. DETECT CONFLICTS
        # ------------------------------------------------

        conflicts = (
            self.conflict_detector.detect(
                results
            )
        )

        # ------------------------------------------------
        # 4. FUSION
        # ------------------------------------------------

        fusion = self.fusion.fuse(
            results
        )

        # ------------------------------------------------
        # 5. FINAL OUTPUT
        # ------------------------------------------------

        return {
            "agents": [
                result.to_dict()
                for result in results
            ],

            "conflicts": conflicts,

            "fusion": fusion
        }
