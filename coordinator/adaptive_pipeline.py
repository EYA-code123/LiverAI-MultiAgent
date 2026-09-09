# =============================================================================
# LiverAI-MultiAgent
# ADAPTIVE COORDINATION PIPELINE
# =============================================================================

from coordinator.liver_coordinator import (
    LiverCoordinator,
)


class AdaptiveCoordinationPipeline:
    """
    Public pipeline for the LiverAI multi-agent system.

    The pipeline delegates the actual intelligence to LiverCoordinator.
    """

    def __init__(
        self,
        agents=None
    ):

        self.coordinator = (
            LiverCoordinator(
                agents=agents
            )
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

        return self.coordinator.register_agent(

            agent_id=agent_id,

            agent=agent,

            task_type=task_type,

            modality=modality
        )

    # =========================================================================
    # UNREGISTER
    # =========================================================================

    def unregister_agent(
        self,
        agent_id
    ):

        return self.coordinator.unregister_agent(
            agent_id
        )

    # =========================================================================
    # LIST
    # =========================================================================

    def list_agents(self):

        return self.coordinator.list_agents()

    # =========================================================================
    # GET
    # =========================================================================

    def get_agent(
        self,
        agent_id
    ):

        return self.coordinator.get_agent(
            agent_id
        )

    # =========================================================================
    # RUN
    # =========================================================================

    def run(
        self,
        patient_id,
        inputs=None,
        images=None,
        ground_truth=None
    ):

        return self.coordinator.run(

            patient_id=patient_id,

            inputs=inputs,

            images=images,

            ground_truth=ground_truth
        )

    # =========================================================================
    # FEEDBACK
    # =========================================================================

    def update_feedback(
        self,
        agent_results,
        ground_truths
    ):

        return self.coordinator.update_feedback(

            agent_results=agent_results,

            ground_truths=ground_truths
        )

    # =========================================================================
    # COMMUNICATION TRACE
    # =========================================================================

    def get_communication_trace(self):

        communication = getattr(
            self.coordinator,
            "communication",
            None
        )

        if communication is None:
            return []

        if hasattr(
            communication,
            "get_trace"
        ):

            return communication.get_trace()

        return []

    # =========================================================================
    # COORDINATION TRACE
    # =========================================================================

    def get_coordination_trace(self):

        return list(
            getattr(
                self.coordinator,
                "coordination_trace",
                []
            )
        )

    # =========================================================================
    # DELEGATION HISTORY
    # =========================================================================

    def get_delegation_history(self):

        return list(
            getattr(
                self.coordinator,
                "delegation_history",
                []
            )
        )

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self):

        return self.coordinator.health_check()


__all__ = [
    "AdaptiveCoordinationPipeline",
]
