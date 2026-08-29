from communication.message import AgentMessage
from coordinator.coordinator import LiverCoordinator


class LiverOrchestrator:

    def __init__(
        self,
        cirrhosis_agent=None,
        fatty_liver_agent=None,
        clinical_agent=None,
        fibrosis_agent=None,
        tumor_agent=None,
        segmentation_agent=None
    ):

        self.agents = {

            "CirrhosisAgent":
                cirrhosis_agent,

            "FattyLiverAgent":
                fatty_liver_agent,

            "ClinicalAgent":
                clinical_agent,

            "FibrosisAgent":
                fibrosis_agent,

            "TumorAgent":
                tumor_agent,

            "SegmentationAgent":
                segmentation_agent
        }

        self.coordinator = LiverCoordinator()

    # =========================================================
    # REGISTER HISTORICAL PERFORMANCE
    # =========================================================

    def register_agent_performance(
        self,
        agent_id,
        performance
    ):

        self.coordinator.register_agent_performance(
            agent_id,
            performance
        )

    # =========================================================
    # EXECUTE ONE AGENT
    # =========================================================

    def execute_agent(
        self,
        agent_id,
        input_data,
        patient_id
    ):

        agent = self.agents.get(
            agent_id
        )

        if agent is None:

            return AgentMessage(
                patient_id=patient_id,
                agent_id=agent_id,
                prediction=None,
                probability=None,
                confidence=0.0,
                uncertainty=1.0,
                quality=0.0,
                details={},
                error="Agent not available"
            )

        try:

            result = agent.predict(
                input_data
            )

            return AgentMessage(
                patient_id=patient_id,
                agent_id=agent_id,
                prediction=result.get(
                    "prediction"
                ),
                probability=result.get(
                    "probability"
                ),
                confidence=result.get(
                    "confidence",
                    0.0
                ),
                uncertainty=result.get(
                    "uncertainty",
                    1.0
                ),
                quality=result.get(
                    "quality",
                    1.0
                ),
                details=result.get(
                    "details",
                    {}
                ),
                error=result.get(
                    "error"
                )
            )

        except Exception as e:

            return AgentMessage(
                patient_id=patient_id,
                agent_id=agent_id,
                prediction=None,
                probability=None,
                confidence=0.0,
                uncertainty=1.0,
                quality=0.0,
                details={},
                error=str(e)
            )

    # =========================================================
    # RUN ALL AGENTS
    # =========================================================

    def run(
        self,
        patient_id,
        clinical_data=None,
        image=None,
        volume=None
    ):

        messages = []

        # -----------------------------------------------------
        # CIRRHOSIS
        # -----------------------------------------------------

        if clinical_data is not None:

            message = self.execute_agent(
                "CirrhosisAgent",
                clinical_data,
                patient_id
            )

            messages.append(message)

        # -----------------------------------------------------
        # FATTY LIVER
        # -----------------------------------------------------

        if clinical_data is not None:

            message = self.execute_agent(
                "FattyLiverAgent",
                clinical_data,
                patient_id
            )

            messages.append(message)

        # -----------------------------------------------------
        # CLINICAL
        # -----------------------------------------------------

        if clinical_data is not None:

            message = self.execute_agent(
                "ClinicalAgent",
                clinical_data,
                patient_id
            )

            messages.append(message)

        # -----------------------------------------------------
        # FIBROSIS
        # -----------------------------------------------------

        if clinical_data is not None:

            message = self.execute_agent(
                "FibrosisAgent",
                clinical_data,
                patient_id
            )

            messages.append(message)

        # -----------------------------------------------------
        # TUMOR
        # -----------------------------------------------------

        if image is not None:

            message = self.execute_agent(
                "TumorAgent",
                image,
                patient_id
            )

            messages.append(message)

        # -----------------------------------------------------
        # SEGMENTATION
        # -----------------------------------------------------

        if volume is not None:

            message = self.execute_agent(
                "SegmentationAgent",
                volume,
                patient_id
            )

            messages.append(message)

        # -----------------------------------------------------
        # COORDINATION
        # -----------------------------------------------------

        coordination = (
            self.coordinator.coordinate(
                messages
            )
        )

        return {
            "patient_id": patient_id,

            "agent_messages": [
                message.to_dict()
                for message in messages
            ],

            "coordination": coordination
        }
