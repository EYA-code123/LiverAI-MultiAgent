import traceback
from datetime import datetime


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

        # ==========================================================
        # AGENTS
        # ==========================================================

        self.fatty_agent = fatty_agent
        self.fibrosis_agent = fibrosis_agent
        self.cirrhosis_agent = cirrhosis_agent
        self.tumor_agent = tumor_agent
        self.segmentation_agent = segmentation_agent
        self.clinical_reasoning_agent = clinical_reasoning_agent

        # ==========================================================
        # REGISTRY
        # ==========================================================

        self.agents = {
            "fatty_liver": self.fatty_agent,
            "fibrosis": self.fibrosis_agent,
            "cirrhosis": self.cirrhosis_agent,
            "tumor_classification": self.tumor_agent,
            "liver_segmentation": self.segmentation_agent,
            "clinical_reasoning": self.clinical_reasoning_agent
        }

        # Remove agents that are not available
        self.agents = {
            name: agent
            for name, agent in self.agents.items()
            if agent is not None
        }

        # ==========================================================
        # STATE
        # ==========================================================

        self.last_results = {}
        self.last_assessment = None
        self.execution_log = []

        print("=" * 80)
        print("LIVERAI MULTI-AGENT SYSTEM")
        print("=" * 80)

        print("\nRegistered Agents:")

        for name in self.agents:
            print(f"  ✓ {name}")

        print("\n✓ Orchestrator initialized")
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

        if agent is None:

            self._log(
                f"[{agent_name.upper()}] "
                "Agent unavailable"
            )

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
                "status": "not_available",
                "prediction": None,
                "probability": None
            }

        try:

            # ==================================================
            # PREDICT
            # ==================================================

            if hasattr(agent, "predict"):

                result = agent.predict(input_data)

            elif hasattr(agent, "analyze"):

                result = agent.analyze(input_data)

            else:

                raise AttributeError(
                    f"{agent_name} has no "
                    "predict() or analyze()"
                )

            # ==================================================
            # NORMALIZE RESULT
            # ==================================================

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
                f"[{agent_name.upper()}] ✗ Error: {e}"
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
    # MAIN PREDICTION
    # ==========================================================

    def predict(
        self,
        clinical_data=None,
        ultrasound_image=None,
        mri_image=None,
        liver_volume=None
    ):

        self.execution_log = []

        self._log("\n" + "=" * 80)
        self._log("STARTING LIVERAI ANALYSIS")
        self._log("=" * 80)

        results = {}

        # ======================================================
        # 1. FATTY LIVER
        # ======================================================

        if self.fatty_agent is not None:

            results["fatty_liver"] = self._run_agent(
                "fatty_liver",
                self.fatty_agent,
                clinical_data
            )

        # ======================================================
        # 2. FIBROSIS
        # ======================================================

        if self.fibrosis_agent is not None:

            results["fibrosis"] = self._run_agent(
                "fibrosis",
                self.fibrosis_agent,
                clinical_data
            )

        # ======================================================
        # 3. CIRRHOSIS
        # ======================================================

        if self.cirrhosis_agent is not None:

            results["cirrhosis"] = self._run_agent(
                "cirrhosis",
                self.cirrhosis_agent,
                clinical_data
            )

        # ======================================================
        # 4. TUMOR CLASSIFICATION
        # ======================================================

        if self.tumor_agent is not None:

            results["tumor_classification"] = self._run_agent(
                "tumor_classification",
                self.tumor_agent,
                mri_image
            )

        # ======================================================
        # 5. LIVER SEGMENTATION
        # ======================================================

        if self.segmentation_agent is not None:

            results["liver_segmentation"] = self._run_agent(
                "liver_segmentation",
                self.segmentation_agent,
                liver_volume
            )

        # ======================================================
        # 6. CLINICAL REASONING
        # ======================================================

        if self.clinical_reasoning_agent is not None:

            results["clinical_reasoning"] = self._run_agent(
                "clinical_reasoning",
                self.clinical_reasoning_agent,
                results
            )

        # ======================================================
        # SAVE STATE
        # ======================================================

        self.last_results = results

        self.last_assessment = results.get(
            "clinical_reasoning"
        )

        self._log("\n" + "=" * 80)
        self._log("LIVERAI ANALYSIS COMPLETED")
        self._log("=" * 80)

        return results

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self):

        return {
            name: {
                "available": agent is not None,
                "class": (
                    agent.__class__.__name__
                    if agent is not None
                    else None
                )
            }
            for name, agent in {
                "fatty_liver": self.fatty_agent,
                "fibrosis": self.fibrosis_agent,
                "cirrhosis": self.cirrhosis_agent,
                "tumor_classification": self.tumor_agent,
                "liver_segmentation": self.segmentation_agent,
                "clinical_reasoning": self.clinical_reasoning_agent
            }.items()
        }
