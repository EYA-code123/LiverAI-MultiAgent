"""
LiverAI Central Orchestrator

Coordinates the specialized liver disease agents
and combines their results into a unified response.
"""


class LiverOrchestrator:

    def __init__(
        self,
        cirrhosis_agent=None,
        fatty_liver_agent=None,
        clinical_reasoning_agent=None,
        fibrosis_agent=None,
        tumor_agent=None,
        segmentation_agent=None
    ):
        """
        Initialize the central orchestrator.

        Each parameter represents one specialized agent.
        """

        self.name = "LiverAI-Orchestrator"

        self.agents = {
            "cirrhosis": cirrhosis_agent,
            "fatty_liver": fatty_liver_agent,
            "clinical_reasoning": clinical_reasoning_agent,
            "fibrosis": fibrosis_agent,
            "tumor": tumor_agent,
            "segmentation": segmentation_agent
        }

    def run_tabular_agents(self, patient_data):
        """
        Run agents that work with tabular/clinical patient data.
        """

        results = {}

        if self.agents["cirrhosis"] is not None:
            results["cirrhosis"] = (
                self.agents["cirrhosis"].predict(patient_data)
            )

        if self.agents["fatty_liver"] is not None:
            results["fatty_liver"] = (
                self.agents["fatty_liver"].predict(patient_data)
            )

        if self.agents["clinical_reasoning"] is not None:
            results["clinical_reasoning"] = (
                self.agents["clinical_reasoning"].predict(patient_data)
            )

        if self.agents["fibrosis"] is not None:
            results["fibrosis"] = (
                self.agents["fibrosis"].predict(patient_data)
            )

        return results

    def run_imaging_agents(
        self,
        image_tensor=None,
        volume=None
    ):
        """
        Run imaging-related agents.

        image_tensor:
            MRI/image input for the tumor classification agent.

        volume:
            3D medical image volume for the segmentation agent.
        """

        results = {}

        if (
            self.agents["tumor"] is not None
            and image_tensor is not None
        ):
            results["tumor"] = (
                self.agents["tumor"].predict(image_tensor)
            )

        if (
            self.agents["segmentation"] is not None
            and volume is not None
        ):
            results["segmentation"] = (
                self.agents["segmentation"].predict(volume)
            )

        return results

    def run(
        self,
        patient_data=None,
        image_tensor=None,
        volume=None
    ):
        """
        Execute the complete LiverAI multi-agent pipeline.
        """

        final_results = {
            "orchestrator": self.name,
            "status": "started",
            "tabular_results": {},
            "imaging_results": {}
        }

        # Run clinical/tabular agents
        if patient_data is not None:

            final_results["tabular_results"] = (
                self.run_tabular_agents(patient_data)
            )

        # Run imaging agents
        if image_tensor is not None or volume is not None:

            final_results["imaging_results"] = (
                self.run_imaging_agents(
                    image_tensor=image_tensor,
                    volume=volume
                )
            )

        final_results["status"] = "completed"

        return final_results

    def get_summary(self, results):
        """
        Create a simple summary of the agents' results.
        """

        summary = {
            "orchestrator": self.name,
            "agents_completed": [],
            "results": results
        }

        for category in [
            "tabular_results",
            "imaging_results"
        ]:

            for agent_name in results.get(category, {}):

                summary["agents_completed"].append(
                    agent_name
                )

        return summary
