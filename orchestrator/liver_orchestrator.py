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

        self.last_results = {}

        self.last_assessment = None

        self.execution_log = []

        print("=" * 80)
        print("LIVERAI ORCHESTRATOR")
        print("=" * 80)

        for name, agent in self.agents.items():

            status = (
                "READY"
                if agent is not None
                else "NOT AVAILABLE"
            )

            print(
                f"{name:<25} : {status}"
            )

        print("=" * 80)

    # ==========================================================
    # LOG
    # ==========================================================

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

    # ==========================================================
    # RUN AGENT
    # ==========================================================

    def _run_agent(
        self,
        name,
        agent,
        data
    ):

        if agent is None:

            return {

                "agent":
                    name,

                "status":
                    "not_available",

                "prediction":
                    None
            }

        if data is None:

            return {

                "agent":
                    name,

                "status":
                    "skipped",

                "prediction":
                    None,

                "reason":
                    "No input provided"
            }

        self._log(
            f"[{name}] START"
        )

        try:

            if hasattr(
                agent,
                "predict"
            ):

                result = agent.predict(
                    data
                )

            elif hasattr(
                agent,
                "analyze"
            ):

                result = agent.analyze(
                    data
                )

            else:

                raise AttributeError(
                    f"{name} does not "
                    "implement predict() "
                    "or analyze()"
                )

            if result is None:

                result = {}

            if not isinstance(
                result,
                dict
            ):

                result = {
                    "prediction": result
                }

            result.setdefault(
                "agent",
                name
            )

            result.setdefault(
                "status",
                "completed"
            )

            self._log(
                f"[{name}] ✓ COMPLETED"
            )

            return result

        except Exception as e:

            self._log(
                f"[{name}] ✗ ERROR: {e}"
            )

            traceback.print_exc()

            return {

                "agent":
                    name,

                "status":
                    "error",

                "prediction":
                    None,

                "error":
                    str(e)
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

        # ------------------------------------------------------
        # 1 FATty LIVER
        # ------------------------------------------------------

        results["fatty_liver"] = (
            self._run_agent(
                "fatty_liver",
                self.fatty_agent,
                clinical_data
            )
        )

        # ------------------------------------------------------
        # 2 FIBROSIS
        # ------------------------------------------------------

        results["fibrosis"] = (
            self._run_agent(
                "fibrosis",
                self.fibrosis_agent,
                clinical_data
            )
        )

        # ------------------------------------------------------
        # 3 CIRRHOSIS
        # ------------------------------------------------------

        results["cirrhosis"] = (
            self._run_agent(
                "cirrhosis",
                self.cirrhosis_agent,
                clinical_data
            )
        )

        # ------------------------------------------------------
        # 4 TUMOR
        # ------------------------------------------------------

        results[
            "tumor_classification"
        ] = self._run_agent(
            "tumor_classification",
            self.tumor_agent,
            tumor_image
        )

        # ------------------------------------------------------
        # 5 SEGMENTATION
        # ------------------------------------------------------

        results[
            "liver_segmentation"
        ] = self._run_agent(
            "liver_segmentation",
            self.segmentation_agent,
            liver_volume
        )

        # ------------------------------------------------------
        # 6 CLINICAL REASONING
        # ------------------------------------------------------

        if self.clinical_reasoning_agent:

            results[
                "clinical_reasoning"
            ] = self._run_agent(

                "clinical_reasoning",

                self.clinical_reasoning_agent,

                results
            )

        else:

            results[
                "clinical_reasoning"
            ] = {

                "agent":
                    "clinical_reasoning",

                "status":
                    "not_available"
            }

        # ------------------------------------------------------
        # SAVE STATE
        # ------------------------------------------------------

        self.last_results = results

        self.last_assessment = (
            results.get(
                "clinical_reasoning"
            )
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

        return results

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

            for name, agent
            in self.agents.items()
        }

    # ==========================================================
    # UNIFIED ASSESSMENT
    # ==========================================================

    def get_unified_assessment(self):

        return self.last_assessment
