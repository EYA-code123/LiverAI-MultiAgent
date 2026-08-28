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

        # ==================================================
        # AGENTS
        # ==================================================

        self.fatty_agent = fatty_agent
        self.fibrosis_agent = fibrosis_agent
        self.cirrhosis_agent = cirrhosis_agent
        self.tumor_agent = tumor_agent
        self.segmentation_agent = segmentation_agent
        self.clinical_reasoning_agent = (
            clinical_reasoning_agent
        )

        # ==================================================
        # REGISTRY
        # ==================================================

        self.agents = {

            "fatty_liver":
                self.fatty_agent,

            "fibrosis":
                self.fibrosis_agent,

            "cirrhosis":
                self.cirrhosis_agent,

            "tumor_classification":
                self.tumor_agent,

            "liver_segmentation":
                self.segmentation_agent,

            "clinical_reasoning":
                self.clinical_reasoning_agent
        }

        # ==================================================
        # STATE
        # ==================================================

        self.last_results = {}

        self.last_assessment = None

        self.execution_log = []

        print(
            "\n"
            + "=" * 80
        )

        print(
            "LIVERAI MULTI-AGENT ORCHESTRATOR"
        )

        print(
            "=" * 80
        )

        print(
            "\nRegistered agents:"
        )

        for name, agent in self.agents.items():

            status = (
                "READY"
                if agent is not None
                else "NOT AVAILABLE"
            )

            print(
                f"  {'✓' if agent else '✗'} "
                f"{name:<25} {status}"
            )

        print(
            "=" * 80
        )

    # ==================================================
    # LOGGING
    # ==================================================

    def _log(self, message):

        timestamp = (
            datetime.now()
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        self.execution_log.append({

            "timestamp":
                timestamp,

            "message":
                message
        })

        print(message)

    # ==================================================
    # RUN ONE AGENT
    # ==================================================

    def _run_agent(
        self,
        agent_name,
        agent,
        input_data
    ):

        self._log(
            f"\n[{agent_name.upper()}] START"
        )

        # --------------------------------------------------
        # AGENT NOT AVAILABLE
        # --------------------------------------------------

        if agent is None:

            self._log(
                f"[{agent_name.upper()}] "
                "NOT AVAILABLE"
            )

            return {

                "agent":
                    agent_name,

                "status":
                    "not_available",

                "prediction":
                    None,

                "probability":
                    None
            }

        # --------------------------------------------------
        # INPUT NOT AVAILABLE
        # --------------------------------------------------

        if input_data is None:

            self._log(
                f"[{agent_name.upper()}] "
                "NO INPUT → SKIPPED"
            )

            return {

                "agent":
                    agent_name,

                "status":
                    "not_available",

                "prediction":
                    None,

                "probability":
                    None
            }

        # --------------------------------------------------
        # EXECUTION
        # --------------------------------------------------

        try:

            if hasattr(
                agent,
                "predict"
            ):

                result = agent.predict(
                    input_data
                )

            else:

                raise AttributeError(
                    f"{agent_name} "
                    "does not implement predict()"
                )

            # --------------------------------------------------
            # NORMALIZE
            # --------------------------------------------------

            if result is None:

                result = {}

            if not isinstance(
                result,
                dict
            ):

                result = {
                    "prediction":
                        result
                }

            result.setdefault(
                "agent",
                agent_name
            )

            result.setdefault(
                "status",
                "completed"
            )

            self._log(
                f"[{agent_name.upper()}] "
                "✓ COMPLETED"
            )

            return result

        except Exception as e:

            self._log(
                f"[{agent_name.upper()}] "
                f"✗ ERROR: {e}"
            )

            traceback.print_exc()

            return {

                "agent":
                    agent_name,

                "status":
                    "error",

                "prediction":
                    None,

                "probability":
                    None,

                "error":
                    str(e)
            }

    # ==================================================
    # MAIN PIPELINE
    # ==================================================

    def predict(
        self,
        clinical_data=None,
        ultrasound_image=None,
        mri_image=None,
        liver_volume=None
    ):

        self.execution_log = []

        self._log(
            "\n"
            + "=" * 80
        )

        self._log(
            "STARTING LIVERAI ANALYSIS"
        )

        self._log(
            "=" * 80
        )

        results = {}

        # ==================================================
        # STAGE 1
        # SPECIALIZED AGENTS
        # ==================================================

        self._log(
            "\n"
            "STAGE 1 — SPECIALIZED AGENTS"
        )

        # --------------------------------------------------
        # FAT
        # --------------------------------------------------

        results[
            "fatty_liver"
        ] = self._run_agent(

            "fatty_liver",

            self.fatty_agent,

            clinical_data
        )

        # --------------------------------------------------
        # FIBROSIS
        # --------------------------------------------------

        results[
            "fibrosis"
        ] = self._run_agent(

            "fibrosis",

            self.fibrosis_agent,

            clinical_data
        )

        # --------------------------------------------------
        # CIRRHOSIS
        # --------------------------------------------------

        results[
            "cirrhosis"
        ] = self._run_agent(

            "cirrhosis",

            self.cirrhosis_agent,

            clinical_data
        )

        # --------------------------------------------------
        # TUMOR
        # --------------------------------------------------

        results[
            "tumor_classification"
        ] = self._run_agent(

            "tumor_classification",

            self.tumor_agent,

            mri_image
        )

        # --------------------------------------------------
        # SEGMENTATION
        # --------------------------------------------------

        results[
            "liver_segmentation"
        ] = self._run_agent(

            "liver_segmentation",

            self.segmentation_agent,

            liver_volume
        )

        # ==================================================
        # STAGE 2
        # CLINICAL REASONING
        # ==================================================

        self._log(
            "\n"
            "STAGE 2 — CLINICAL REASONING"
        )

        results[
            "clinical_reasoning"
        ] = self._run_agent(

            "clinical_reasoning",

            self.clinical_reasoning_agent,

            results
        )

        # ==================================================
        # UNIFIED ASSESSMENT
        # ==================================================

        clinical_result = results.get(
            "clinical_reasoning",
            {}
        )

        unified_assessment = (
            clinical_result.get(
                "unified_assessment"
            )
        )

        # ==================================================
        # FINAL OUTPUT
        # ==================================================

        final_result = {

            "system":
                self.name,

            "timestamp":
                datetime.now().isoformat(),

            "status":
                "completed",

            "agent_results":
                results,

            "unified_assessment":
                unified_assessment,

            "execution_log":
                self.execution_log
        }

        # ==================================================
        # SAVE STATE
        # ==================================================

        self.last_results = results

        self.last_assessment = (
            unified_assessment
        )

        self._log(
            "\n"
            + "=" * 80
        )

        self._log(
            "LIVERAI ANALYSIS COMPLETED"
        )

        self._log(
            "=" * 80
        )

        return final_result

    # ==================================================
    # STATUS
    # ==================================================

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

            for name, agent in self.agents.items()
        }
