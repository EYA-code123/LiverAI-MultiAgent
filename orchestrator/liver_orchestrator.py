# ==============================================================
# LiverAI-MultiAgent
# Complete Multi-Agent Orchestrator
#
# Agents:
#   1. Fatty Liver Agent
#   2. Fibrosis Agent
#   3. Cirrhosis Agent
#   4. Tumor Classification Agent
#   5. Liver Segmentation Agent
#   6. Clinical Reasoning Agent
#
# Architecture:
#
# Patient / Imaging Data
#          |
#          v
#   LiverAI Orchestrator
#          |
#   +------+------+------+------+------+
#   |      |      |      |      |      |
# Fatty  Fibro  Cirrho Tumor  Segm  ...
#   |      |      |      |      |
#   +------+------+------+------+------+
#          |
#          v
#   Clinical Reasoning
#          |
#          v
#   Unified Liver Assessment
# ==============================================================


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

        self.name = "LiverAI Orchestrator"

        # ======================================================
        # SPECIALIZED AGENTS
        # ======================================================

        self.fatty_agent = fatty_agent

        self.fibrosis_agent = fibrosis_agent

        self.cirrhosis_agent = cirrhosis_agent

        self.tumor_agent = tumor_agent

        self.segmentation_agent = segmentation_agent

        # ======================================================
        # REASONING AGENT
        # ======================================================

        self.clinical_reasoning_agent = clinical_reasoning_agent

        # ======================================================
        # AGENT REGISTRY
        # ======================================================

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

        # ======================================================
        # SYSTEM STATE
        # ======================================================

        self.last_results = {}

        self.last_assessment = None

        self.execution_log = []

        print("=" * 80)
        print("LIVERAI MULTI-AGENT SYSTEM")
        print("=" * 80)

        print("\nRegistered Agents:")

        for agent_name in self.agents:

            print(f"  ✓ {agent_name}")

        print("\n✓ LiverAI Orchestrator initialized")
        print("=" * 80)

    # ==========================================================
    # LOGGING
    # ==========================================================

    def _log(self, message):

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        entry = {
            "timestamp": timestamp,
            "message": message
        }

        self.execution_log.append(entry)

        print(message)

    # ==========================================================
    # SAFE AGENT EXECUTION
    # ==========================================================

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
                "No input available → skipped."
            )

            return {
                "agent": agent_name,
                "status": "not_available",
                "prediction": None,
                "confidence": None
            }

        try:

            # --------------------------------------------------
            # STANDARD PREDICT INTERFACE
            # --------------------------------------------------

            if hasattr(agent, "predict"):

                result = agent.predict(input_data)

            # --------------------------------------------------
            # ALTERNATIVE ANALYZE INTERFACE
            # --------------------------------------------------

            elif hasattr(agent, "analyze"):

                result = agent.analyze(input_data)

            else:

                raise AttributeError(
                    f"{agent_name} does not have "
                    "`predict()` or `analyze()`."
                )

            # --------------------------------------------------
            # NORMALIZE RESULT
            # --------------------------------------------------

            if result is None:

                result = {}

            if not isinstance(result, dict):

                result = {
                    "prediction": result
                }

            result["agent"] = agent_name

            result.setdefault(
                "status",
                "success"
            )

            self._log(
                f"[{agent_name.upper()}] ✓ Completed"
            )

            return result

        except Exception as e:

            self._log(
                f"[{agent_name.upper()}] ✗ Error: {e}"
            )

            return {

                "agent": agent_name,

                "status": "error",

                "prediction": None,

                "confidence": None,

                "error": str(e),

                "traceback":
                    traceback.format_exc()
            }

    # ==========================================================
    # RUN FATty LIVER AGENT
    # ==========================================================

    def run_fatty_liver(
        self,
        clinical_data
    ):

        return self._run_agent(
            "fatty_liver",
            self.fatty_agent,
            clinical_data
        )

    # ==========================================================
    # RUN FIBROSIS AGENT
    # ==========================================================

    def run_fibrosis(
        self,
        ultrasound_image
    ):

        return self._run_agent(
            "fibrosis",
            self.fibrosis_agent,
            ultrasound_image
        )

    # ==========================================================
    # RUN CIRRHOSIS AGENT
    # ==========================================================

    def run_cirrhosis(
        self,
        clinical_data
    ):

        return self._run_agent(
            "cirrhosis",
            self.cirrhosis_agent,
            clinical_data
        )

    # ==========================================================
    # RUN TUMOR CLASSIFICATION AGENT
    # ==========================================================

    def run_tumor_classification(
        self,
        mri_image
    ):

        return self._run_agent(
            "tumor_classification",
            self.tumor_agent,
            mri_image
        )

    # ==========================================================
    # RUN LIVER SEGMENTATION AGENT
    # ==========================================================

    def run_liver_segmentation(
        self,
        liver_volume
    ):

        return self._run_agent(
            "liver_segmentation",
            self.segmentation_agent,
            liver_volume
        )

    # ==========================================================
    # RUN ALL SPECIALIZED AGENTS
    # ==========================================================

    def run_specialized_agents(
        self,
        clinical_data=None,
        ultrasound_image=None,
        mri_image=None,
        liver_volume=None
    ):

        self._log("\n")
        self._log("=" * 80)
        self._log("RUNNING SPECIALIZED AGENTS")
        self._log("=" * 80)

        results = {}

        # ------------------------------------------------------
        # 1. FATTY LIVER
        # ------------------------------------------------------

        self._log(
            "\n[1/5] Fatty Liver Agent"
        )

        results["fatty_liver"] = \
            self.run_fatty_liver(
                clinical_data
            )

        # ------------------------------------------------------
        # 2. FIBROSIS
        # ------------------------------------------------------

        self._log(
            "\n[2/5] Fibrosis Agent"
        )

        results["fibrosis"] = \
            self.run_fibrosis(
                ultrasound_image
            )

        # ------------------------------------------------------
        # 3. CIRRHOSIS
        # ------------------------------------------------------

        self._log(
            "\n[3/5] Cirrhosis Agent"
        )

        results["cirrhosis"] = \
            self.run_cirrhosis(
                clinical_data
            )

        # ------------------------------------------------------
        # 4. TUMOR
        # ------------------------------------------------------

        self._log(
            "\n[4/5] Tumor Classification Agent"
        )

        results["tumor_classification"] = \
            self.run_tumor_classification(
                mri_image
            )

        # ------------------------------------------------------
        # 5. SEGMENTATION
        # ------------------------------------------------------

        self._log(
            "\n[5/5] Liver Segmentation Agent"
        )

        results["liver_segmentation"] = \
            self.run_liver_segmentation(
                liver_volume
            )

        # ------------------------------------------------------
        # SAVE RESULTS
        # ------------------------------------------------------

        self.last_results = results

        return results

    # ==========================================================
    # CLINICAL REASONING
    # ==========================================================

    def run_clinical_reasoning(
        self,
        agent_results
    ):

        self._log("\n")
        self._log("=" * 80)
        self._log("CLINICAL REASONING")
        self._log("=" * 80)

        try:

            # --------------------------------------------------
            # Preferred interface
            # --------------------------------------------------

            if hasattr(
                self.clinical_reasoning_agent,
                "analyze"
            ):

                reasoning = \
                    self.clinical_reasoning_agent.analyze(
                        agent_results
                    )

            # --------------------------------------------------
            # Alternative interface
            # --------------------------------------------------

            elif hasattr(
                self.clinical_reasoning_agent,
                "predict"
            ):

                reasoning = \
                    self.clinical_reasoning_agent.predict(
                        agent_results
                    )

            else:

                raise AttributeError(
                    "Clinical Reasoning Agent must have "
                    "`analyze()` or `predict()`."
                )

            if reasoning is None:

                reasoning = {}

            self._log(
                "✓ Clinical Reasoning completed"
            )

            return reasoning

        except Exception as e:

            self._log(
                f"✗ Clinical Reasoning error: {e}"
            )

            return {

                "status": "error",

                "error": str(e),

                "traceback":
                    traceback.format_exc()
            }

    # ==========================================================
    # BUILD UNIFIED ASSESSMENT
    # ==========================================================

    def build_unified_assessment(
        self,
        agent_results,
        clinical_reasoning
    ):

        assessment = {

            "system": "LiverAI-MultiAgent",

            "timestamp":
                datetime.now().isoformat(),

            "agents": {

                "fatty_liver":
                    agent_results.get(
                        "fatty_liver"
                    ),

                "fibrosis":
                    agent_results.get(
                        "fibrosis"
                    ),

                "cirrhosis":
                    agent_results.get(
                        "cirrhosis"
                    ),

                "tumor_classification":
                    agent_results.get(
                        "tumor_classification"
                    ),

                "liver_segmentation":
                    agent_results.get(
                        "liver_segmentation"
                    )
            },

            "clinical_reasoning":
                clinical_reasoning,

            "system_status":
                "completed"
        }

        self.last_assessment = assessment

        return assessment

    # ==========================================================
    # COMPLETE PIPELINE
    # ==========================================================

    def predict(
        self,
        clinical_data=None,
        ultrasound_image=None,
        mri_image=None,
        liver_volume=None
    ):

        self.execution_log = []

        self._log("\n")
        self._log("=" * 80)
        self._log("LIVERAI MULTI-AGENT ANALYSIS")
        self._log("=" * 80)

        # ======================================================
        # STEP 1
        # ======================================================

        self._log(
            "\nSTEP 1/3 → Specialized Agents"
        )

        agent_results = \
            self.run_specialized_agents(

                clinical_data=clinical_data,

                ultrasound_image=ultrasound_image,

                mri_image=mri_image,

                liver_volume=liver_volume
            )

        # ======================================================
        # STEP 2
        # ======================================================

        self._log(
            "\nSTEP 2/3 → Clinical Reasoning"
        )

        clinical_reasoning = \
            self.run_clinical_reasoning(
                agent_results
            )

        # ======================================================
        # STEP 3
        # ======================================================

        self._log(
            "\nSTEP 3/3 → Unified Assessment"
        )

        final_assessment = \
            self.build_unified_assessment(

                agent_results,

                clinical_reasoning
            )

        self._log(
            "\n✓ Unified Liver Assessment generated"
        )

        self._log(
            "=" * 80
        )

        return final_assessment

    # ==========================================================
    # GET LAST RESULTS
    # ==========================================================

    def get_last_results(self):

        return self.last_results

    # ==========================================================
    # GET LAST ASSESSMENT
    # ==========================================================

    def get_last_assessment(self):

        return self.last_assessment

    # ==========================================================
    # GET EXECUTION LOG
    # ==========================================================

    def get_execution_log(self):

        return self.execution_log

    # ==========================================================
    # SYSTEM STATUS
    # ==========================================================

    def get_system_status(self):

        status = {}

        for name, agent in self.agents.items():

            status[name] = {

                "loaded":
                    agent is not None,

                "class":
                    agent.__class__.__name__
            }

        return status
