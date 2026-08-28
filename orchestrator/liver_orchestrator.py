from datetime import datetime
import traceback


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

        self.name = "LiverAI Orchestrator"

        self.fatty_agent = fatty_agent
        self.fibrosis_agent = fibrosis_agent
        self.cirrhosis_agent = cirrhosis_agent
        self.tumor_agent = tumor_agent
        self.segmentation_agent = segmentation_agent
        self.clinical_reasoning_agent = (
            clinical_reasoning_agent
        )

        self.last_results = {}
        self.last_assessment = None
        self.execution_log = []

        self.agents = {
            "fatty_liver": self.fatty_agent,
            "fibrosis": self.fibrosis_agent,
            "cirrhosis": self.cirrhosis_agent,
            "tumor": self.tumor_agent,
            "segmentation": self.segmentation_agent,
            "clinical_reasoning":
                self.clinical_reasoning_agent
        }

        self.agents = {
            name: agent
            for name, agent in self.agents.items()
            if agent is not None
        }

        print("=" * 80)
        print("LIVERAI ORCHESTRATOR")
        print("=" * 80)

        for name in self.agents:
            print(f"✓ Registered: {name}")

        print("=" * 80)

    # ==========================================================
    # LOGGING
    # ==========================================================

    def _log(self, message):

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        self.execution_log.append({
            "timestamp": timestamp,
            "message": message
        })

        print(message)

    # ==========================================================
    # SAFE EXECUTION
    # ==========================================================

    def _execute(
        self,
        agent_name,
        agent,
        input_data
    ):

        self._log(
            f"[{agent_name.upper()}] Starting..."
        )

        if agent is None:

            return {
                "agent": agent_name,
                "status": "not_available",
                "prediction": None,
                "probability": None
            }

        if input_data is None:

            self._log(
                f"[{agent_name.upper()}] "
                "No input → skipped"
            )

            return {
                "agent": agent_name,
                "status": "skipped",
                "prediction": None,
                "probability": None
            }

        try:

            if hasattr(agent, "predict"):

                result = agent.predict(input_data)

            elif hasattr(agent, "analyze"):

                result = agent.analyze(input_data)

            else:

                raise AttributeError(
                    f"{agent_name} has neither "
                    "predict() nor analyze()"
                )

            if result is None:

                result = {}

            if not isinstance(result, dict):

                result = {
                    "prediction": result
                }

            result["agent"] = agent_name

            result.setdefault(
                "status",
                "completed"
            )

            self._log(
                f"[{agent_name.upper()}] ✓ Completed"
            )

            return result

        except Exception as e:

            self._log(
                f"[{agent_name.upper()}] ✗ ERROR: {e}"
            )

            traceback.print_exc()

            return {
                "agent": agent_name,
                "status": "error",
                "prediction": None,
                "probability": None,
                "error": str(e)
            }

    # ==========================================================
    # VALIDATE INPUTS
    # ==========================================================

    def validate_inputs(
        self,
        clinical_data=None,
        tumor_image=None,
        liver_volume=None
    ):

        validation = {
            "clinical_data":
                clinical_data is not None,

            "tumor_image":
                tumor_image is not None,

            "liver_volume":
                liver_volume is not None
        }

        return validation

    # ==========================================================
    # RUN SPECIALIZED AGENTS
    # ==========================================================

    def run_specialized_agents(
        self,
        clinical_data=None,
        tumor_image=None,
        liver_volume=None
    ):

        results = {}

        # ------------------------------------------------------
        # FAT
        # ------------------------------------------------------

        if self.fatty_agent is not None:

            results["fatty_liver"] = self._execute(
                "fatty_liver",
                self.fatty_agent,
                clinical_data
            )

        # ------------------------------------------------------
        # FIBROSIS
        # ------------------------------------------------------

        if self.fibrosis_agent is not None:

            results["fibrosis"] = self._execute(
                "fibrosis",
                self.fibrosis_agent,
                clinical_data
            )

        # ------------------------------------------------------
        # CIRRHOSIS
        # ------------------------------------------------------

        if self.cirrhosis_agent is not None:

            results["cirrhosis"] = self._execute(
                "cirrhosis",
                self.cirrhosis_agent,
                clinical_data
            )

        # ------------------------------------------------------
        # TUMOR
        # ------------------------------------------------------

        if self.tumor_agent is not None:

            results["tumor"] = self._execute(
                "tumor",
                self.tumor_agent,
                tumor_image
            )

        # ------------------------------------------------------
        # SEGMENTATION
        # ------------------------------------------------------

        if self.segmentation_agent is not None:

            results["segmentation"] = self._execute(
                "segmentation",
                self.segmentation_agent,
                liver_volume
            )

        return results

    # ==========================================================
    # CLINICAL REASONING
    # ==========================================================

    def run_clinical_reasoning(
        self,
        specialized_results
    ):

        if self.clinical_reasoning_agent is None:

            return {
                "agent": "clinical_reasoning",
                "status": "not_available"
            }

        self._log(
            "\n[CLINICAL REASONING] "
            "Receiving all agent results..."
        )

        try:

            if hasattr(
                self.clinical_reasoning_agent,
                "analyze"
            ):

                assessment = (
                    self.clinical_reasoning_agent.analyze(
                        specialized_results
                    )
                )

            elif hasattr(
                self.clinical_reasoning_agent,
                "predict"
            ):

                assessment = (
                    self.clinical_reasoning_agent.predict(
                        specialized_results
                    )
                )

            else:

                raise AttributeError(
                    "Clinical Reasoning Agent must "
                    "implement analyze() or predict()"
                )

            if not isinstance(
                assessment,
                dict
            ):

                assessment = {
                    "assessment": assessment
                }

            assessment["agent"] = (
                "clinical_reasoning"
            )

            assessment.setdefault(
                "status",
                "completed"
            )

            return assessment

        except Exception as e:

            traceback.print_exc()

            return {
                "agent": "clinical_reasoning",
                "status": "error",
                "error": str(e)
            }

    # ==========================================================
    # MAIN PIPELINE
    # ==========================================================

    def predict(
        self,
        clinical_data=None,
        tumor_image=None,
        liver_volume=None
    ):

        self.execution_log = []

        self._log("=" * 80)
        self._log("STARTING LIVERAI ANALYSIS")
        self._log("=" * 80)

        # ------------------------------------------------------
        # STEP 1
        # ------------------------------------------------------

        validation = self.validate_inputs(
            clinical_data,
            tumor_image,
            liver_volume
        )

        self._log(
            f"Input validation: {validation}"
        )

        # ------------------------------------------------------
        # STEP 2
        # ------------------------------------------------------

        specialized_results = (
            self.run_specialized_agents(
                clinical_data=clinical_data,
                tumor_image=tumor_image,
                liver_volume=liver_volume
            )
        )

        # ------------------------------------------------------
        # STEP 3
        # ------------------------------------------------------

        clinical_assessment = (
            self.run_clinical_reasoning(
                specialized_results
            )
        )

        # ------------------------------------------------------
        # STEP 4
        # ------------------------------------------------------

        unified_assessment = {

            "system":
                "LiverAI Multi-Agent",

            "timestamp":
                datetime.now().isoformat(),

            "input_validation":
                validation,

            "agent_results":
                specialized_results,

            "clinical_reasoning":
                clinical_assessment
        }

        # ------------------------------------------------------
        # SAVE STATE
        # ------------------------------------------------------

        self.last_results = (
            specialized_results
        )

        self.last_assessment = (
            clinical_assessment
        )

        self._log("=" * 80)
        self._log("LIVERAI ANALYSIS COMPLETED")
        self._log("=" * 80)

        return unified_assessment

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self):

        return {

            name: {

                "available":
                    agent is not None,

                "class":
                    (
                        agent.__class__.__name__
                        if agent is not None
                        else None
                    )

            }

            for name, agent in {
                "fatty_liver":
                    self.fatty_agent,

                "fibrosis":
                    self.fibrosis_agent,

                "cirrhosis":
                    self.cirrhosis_agent,

                "tumor":
                    self.tumor_agent,

                "segmentation":
                    self.segmentation_agent,

                "clinical_reasoning":
                    self.clinical_reasoning_agent

            }.items()
        }
