import traceback
from datetime import datetime


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

        self.name = (
            "LiverAI Orchestrator"
        )

        # ==================================================
        # AGENTS
        # ==================================================

        self.fatty_agent = fatty_agent

        self.fibrosis_agent = fibrosis_agent

        self.cirrhosis_agent = cirrhosis_agent

        self.tumor_agent = tumor_agent

        self.segmentation_agent = (
            segmentation_agent
        )

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

    # ======================================================
    # LOG
    # ======================================================

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

    # ======================================================
    # SAFE AGENT EXECUTION
    # ======================================================

    def _run_agent(
        self,
        agent_name,
        agent,
        input_data
    ):

        self._log(
            f"\n[{agent_name.upper()}] Starting..."
        )

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

                "probability":
                    None,

                "error":
                    str(e),

                "traceback":
                    traceback.format_exc()
            }

    # ======================================================
    # FATTIY LIVER
    # ======================================================

    def run_fatty_liver(
        self,
        clinical_data
    ):

        return self._run_agent(

            "fatty_liver",

            self.fatty_agent,

            clinical_data
        )

    # ======================================================
    # FIBROSIS
    # ======================================================

    def run_fibrosis(
        self,
        clinical_data
    ):

        return self._run_agent(

            "fibrosis",

            self.fibrosis_agent,

            clinical_data
        )

    # ======================================================
    # CIRRHOSIS
    # ======================================================

    def run_cirrhosis(
        self,
        clinical_data
    ):

        return self._run_agent(

            "cirrhosis",

            self.cirrhosis_agent,

            clinical_data
        )

    # ======================================================
    # TUMOR
    # ======================================================

    def run_tumor_classification(
        self,
        mri_image
    ):

        return self._run_agent(

            "tumor_classification",

            self.tumor_agent,

            mri_image
        )

    # ======================================================
    # SEGMENTATION
    # ======================================================

    def run_liver_segmentation(
        self,
        liver_volume
    ):

        return self._run_agent(

            "liver_segmentation",

            self.segmentation_agent,

            liver_volume
        )

    # ======================================================
    # SPECIALIZED AGENTS
    # ======================================================

    def run_specialized_agents(

        self,

        clinical_data=None,

        ultrasound_image=None,

        mri_image=None,

        liver_volume=None

    ):

        self._log(
            "\n"
            + "=" * 80
        )

        self._log(
            "RUNNING SPECIALIZED AGENTS"
        )

        self._log(
            "=" * 80
        )

        results = {}

        # ==================================================
        # 1 — FATTY LIVER
        # ==================================================

        self._log(
            "\n[1/5] Fatty Liver"
        )

        results[
            "fatty_liver"
        ] = self.run_fatty_liver(
            clinical_data
        )

        # ==================================================
        # 2 — FIBROSIS
        # ==================================================

        self._log(
            "\n[2/5] Fibrosis"
        )

        # IMPORTANT:
        # Fibrosis is TABULAR.
        # It receives clinical_data.
        #
        # ultrasound_image is NOT used here.

        results[
            "fibrosis"
        ] = self.run_fibrosis(
            clinical_data
        )

        # ==================================================
        # 3 — CIRRHOSIS
        # ==================================================

        self._log(
            "\n[3/5] Cirrhosis"
        )

        results[
            "cirrhosis"
        ] = self.run_cirrhosis(
            clinical_data
        )

        # ==================================================
        # 4 — TUMOR
        # ==================================================

        self._log(
            "\n[4/5] Tumor Classification"
        )

        results[
            "tumor_classification"
        ] = self.run_tumor_classification(
            mri_image
        )

        # ==================================================
        # 5 — SEGMENTATION
        # ==================================================

        self._log(
            "\n[5/5] Liver Segmentation"
        )

        results[
            "liver_segmentation"
        ] = self.run_liver_segmentation(
            liver_volume
        )

        # ==================================================
        # SAVE STATE
        # ==================================================

        self.last_results = results

        return results

    # ======================================================
    # CLINICAL REASONING
    # ======================================================

    def run_clinical_reasoning(
        self,
        agent_results
    ):

        self._log(
            "\n"
            + "=" * 80
        )

        self._log(
            "CLINICAL REASONING"
        )

        self._log(
            "=" * 80
        )

        try:

            if hasattr(
                self.clinical_reasoning_agent,
                "predict"
            ):

                reasoning = (
                    self.clinical_reasoning_agent
                    .predict(
                        agent_results
                    )
                )

            elif hasattr(
                self.clinical_reasoning_agent,
                "analyze"
            ):

                reasoning = (
                    self.clinical_reasoning_agent
                    .analyze(
                        agent_results
                    )
                )

            else:

                raise AttributeError(
                    "Clinical Reasoning Agent "
                    "must implement predict() "
                    "or analyze()."
                )

            self._log(
                "✓ Clinical Reasoning completed"
            )

            return reasoning

        except Exception as e:

            self._log(
                f"✗ Clinical Reasoning error: "
                f"{e}"
            )

            return {

                "agent":
                    "ClinicalReasoningAgent",

                "status":
                    "error",

                "error":
                    str(e),

                "traceback":
                    traceback.format_exc()
            }

    # ======================================================
    # UNIFIED ASSESSMENT
    # ======================================================

    def build_unified_assessment(

        self,

        agent_results,

        clinical_reasoning

    ):

        return {

            "system":
                "LiverAI-MultiAgent",

            "timestamp":
                datetime.now().isoformat(),

            "agents":
                agent_results,

            "clinical_reasoning":
                clinical_reasoning,

            "system_status":
                "completed"
        }

    # ======================================================
    # COMPLETE PIPELINE
    # ======================================================

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
            "LIVERAI MULTI-AGENT ANALYSIS"
        )

        self._log(
            "=" * 80
        )

        # ==================================================
        # STEP 1
        # ==================================================

        self._log(
            "\nSTEP 1/3 → Specialized Agents"
        )

        agent_results = (
            self.run_specialized_agents(

                clinical_data=
                    clinical_data,

                ultrasound_image=
                    ultrasound_image,

                mri_image=
                    mri_image,

                liver_volume=
                    liver_volume
            )
        )

        # ==================================================
        # STEP 2
        # ==================================================

        self._log(
            "\nSTEP 2/3 → Clinical Reasoning"
        )

        clinical_reasoning = (
            self.run_clinical_reasoning(
                agent_results
            )
        )

        # ==================================================
        # STEP 3
        # ==================================================

        self._log(
            "\nSTEP 3/3 → Unified Assessment"
        )

        final_assessment = (
            self.build_unified_assessment(

                agent_results,

                clinical_reasoning
            )
        )

        self.last_assessment = (
            final_assessment
        )

        self._log(
            "\n✓ Unified assessment generated"
        )

        self._log(
            "=" * 80
        )

        return final_assessment

    # ======================================================
    # GETTERS
    # ======================================================

    def get_last_results(self):

        return self.last_results

    def get_last_assessment(self):

        return self.last_assessment

    def get_execution_log(self):

        return self.execution_log

    def get_system_status(self):

        status = {}

        for name, agent in (
            self.agents.items()
        ):

            status[name] = {

                "loaded":
                    agent is not None,

                "class":
                    agent.__class__.__name__
            }

        return status
