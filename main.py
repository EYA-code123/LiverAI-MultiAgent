# ==============================================================================
# LiverAI Multi-Agent Orchestrator
# ==============================================================================

from datetime import datetime
import traceback


class LiverAIOrchestrator:

    def __init__(
        self,
        fatty_agent,
        fibrosis_agent,
        cirrhosis_agent,
        tumor_agent,
        segmentation_agent,
        clinical_reasoning_agent
    ):

        self.name = "LiverAI Orchestrator"

        # ==========================================================================
        # AGENTS
        # ==========================================================================

        self.fatty_agent = fatty_agent

        self.fibrosis_agent = fibrosis_agent

        self.cirrhosis_agent = cirrhosis_agent

        self.tumor_agent = tumor_agent

        self.segmentation_agent = segmentation_agent

        self.clinical_reasoning_agent = (
            clinical_reasoning_agent
        )

        # ==========================================================================
        # AGENT REGISTRY
        # ==========================================================================

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

        # ==========================================================================
        # STATE
        # ==========================================================================

        self.last_results = {}

        self.last_assessment = None

        self.execution_log = []

        # ==========================================================================
        # DISPLAY
        # ==========================================================================

        print("=" * 80)

        print(
            "LIVERAI MULTI-AGENT SYSTEM"
        )

        print("=" * 80)

        print(
            "\nRegistered Agents:"
        )

        for name in self.agents:

            print(
                f"  ✓ {name}"
            )

        print(
            "\n✓ Orchestrator initialized"
        )

        print("=" * 80)


    # ==========================================================================
    # LOG
    # ==========================================================================

    def _log(
        self,
        message
    ):

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

        print(
            message
        )


    # ==========================================================================
    # SAFE AGENT EXECUTION
    # ==========================================================================

    def _run_agent(
        self,
        agent_name,
        agent,
        input_data
    ):

        self._log(
            f"\n[{agent_name.upper()}] Starting..."
        )

        # ----------------------------------------------------------------------
        # No input
        # ----------------------------------------------------------------------

        if input_data is None:

            self._log(
                f"[{agent_name.upper()}] "
                "No input → skipped"
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

        # ----------------------------------------------------------------------
        # Execute
        # ----------------------------------------------------------------------

        try:

            if hasattr(
                agent,
                "predict"
            ):

                result = agent.predict(
                    input_data
                )

            elif hasattr(
                agent,
                "analyze"
            ):

                result = agent.analyze(
                    input_data
                )

            else:

                raise AttributeError(
                    f"{agent_name} has no "
                    "predict() or analyze()"
                )

            # ------------------------------------------------------------------
            # Normalize result
            # ------------------------------------------------------------------

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

            result["agent"] = (
                agent_name
            )

            result.setdefault(
                "status",
                "completed"
            )

            self._log(
                f"[{agent_name.upper()}] "
                "✓ Completed"
            )

            return result

        except Exception as e:

            self._log(
                f"[{agent_name.upper()}] "
                f"✗ Error: {e}"
            )

            return {

                "agent":
                    agent_name,

                "status":
                    "error",

                "prediction":
                    None,

                "error":
                    str(e),

                "traceback":
                    traceback.format_exc()
            }


    # ==========================================================================
    # PREDICT
    # ==========================================================================

    def predict(
        self,
        clinical_data=None,
        ultrasound_image=None,
        mri_image=None,
        segmentation_volume=None
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

        # ==========================================================================
        # RESULTS
        # ==========================================================================

        results = {}

        # ==========================================================================
        # 1. FATTY LIVER
        # ==========================================================================

        results["fatty_liver"] = (
            self._run_agent(
                "fatty_liver",
                self.fatty_agent,
                clinical_data
            )
        )

        # ==========================================================================
        # 2. FIBROSIS
        # ==========================================================================

        results["fibrosis"] = (
            self._run_agent(
                "fibrosis",
                self.fibrosis_agent,
                clinical_data
            )
        )

        # ==========================================================================
        # 3. CIRRHOSIS
        # ==========================================================================

        results["cirrhosis"] = (
            self._run_agent(
                "cirrhosis",
                self.cirrhosis_agent,
                clinical_data
            )
        )

        # ==========================================================================
        # 4. TUMOR
        # ==========================================================================

        results["tumor_classification"] = (
            self._run_agent(
                "tumor_classification",
                self.tumor_agent,
                mri_image
            )
        )

        # ==========================================================================
        # 5. SEGMENTATION
        # ==========================================================================

        results["liver_segmentation"] = (
            self._run_agent(
                "liver_segmentation",
                self.segmentation_agent,
                segmentation_volume
            )
        )

        # ==========================================================================
        # 6. CLINICAL REASONING
        # ==========================================================================

        self._log(
            "\n[CLINICAL REASONING] "
            "Integrating agent results..."
        )

        try:

            clinical_result = (
                self.clinical_reasoning_agent.predict(
                    results
                )
            )

            if clinical_result is None:

                clinical_result = {}

            if not isinstance(
                clinical_result,
                dict
            ):

                clinical_result = {

                    "prediction":
                        clinical_result
                }

            clinical_result["agent"] = (
                "clinical_reasoning"
            )

            clinical_result.setdefault(
                "status",
                "completed"
            )

            results[
                "clinical_reasoning"
            ] = clinical_result

            self._log(
                "[CLINICAL REASONING] "
                "✓ Completed"
            )

        except Exception as e:

            self._log(
                "[CLINICAL REASONING] "
                f"✗ Error: {e}"
            )

            results[
                "clinical_reasoning"
            ] = {

                "agent":
                    "clinical_reasoning",

                "status":
                    "error",

                "prediction":
                    None,

                "error":
                    str(e)
            }

        # ==========================================================================
        # SAVE STATE
        # ==========================================================================

        self.last_results = results

        self.last_assessment = (
            results.get(
                "clinical_reasoning"
            )
        )

        # ==========================================================================
        # FINAL RESULT
        # ==========================================================================

        final_result = {

            "timestamp":
                datetime.now().isoformat(),

            "status":
                "completed",

            "agents":
                results,

            "clinical_assessment":
                self.last_assessment,

            "execution_log":
                self.execution_log
        }

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


    # ==========================================================================
    # GET LAST RESULTS
    # ==========================================================================

    def get_last_results(self):

        return self.last_results


    # ==========================================================================
    # GET LAST ASSESSMENT
    # ==========================================================================

    def get_last_assessment(self):

        return self.last_assessment


    # ==========================================================================
    # GET EXECUTION LOG
    # ==========================================================================

    def get_execution_log(self):

        return self.execution_log


    # ==========================================================================
    # LIST AGENTS
    # ==========================================================================

    def list_agents(self):

        return list(
            self.agents.keys()
        )
