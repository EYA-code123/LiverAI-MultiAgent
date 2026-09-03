from coordinator.coordinator import (
    LiverAICoordinator
)


class AdaptiveCoordinationPipeline:

    def __init__(
        self,
        agents=None
    ):

        self.coordinator = (
            LiverAICoordinator(
                agents=agents
            )
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

        self.coordinator.register_agent(

            agent_id=
                agent_id,

            agent=
                agent,

            task_type=
                task_type,

            modality=
                modality
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

        return self.coordinator.run(

            patient_id=
                patient_id,

            inputs=
                inputs,

            ground_truth=
                ground_truth
        )

    # =========================================================
    # FEEDBACK
    # =========================================================

    def feedback(
        self,
        agent_results,
        ground_truth
    ):

        return (
            self.coordinator
            .feedback_engine
            .update(

                agent_results=
                    agent_results,

                ground_truth=
                    ground_truth
            )
        )
