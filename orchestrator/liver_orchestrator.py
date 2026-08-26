```python
import traceback
from datetime import datetime


class LiverAIOrchestrator:

    """
    Central coordinator for the LiverAI Multi-Agent system.

    Agents:
        1. Fatty Liver Agent
        2. Fibrosis Agent
        3. Cirrhosis Agent
        4. Tumor Classification Agent
        5. Liver Segmentation Agent
        6. Clinical Reasoning Agent
    """

    def __init__(
        self,
        fatty_agent,
        fibrosis_agent,
        cirrhosis_agent,
        tumor_agent,
        segmentation_agent,
        clinical_reasoning_agent
    ):

        self.fatty_agent = fatty_agent
        self.fibrosis_agent = fibrosis_agent
        self.cirrhosis_agent = cirrhosis_agent
        self.tumor_agent = tumor_agent
        self.segmentation_agent = segmentation_agent
        self.clinical_reasoning_agent = clinical_reasoning_agent

        self.last_results = {}
        self.last_assessment = None
        self.execution_log = []

        self.agents = {
            "fatty_liver": self.fatty_agent,
            "fibrosis": self.fibrosis_agent,
            "cirrhosis": self.cirrhosis_agent,
            "tumor_classification": self.tumor_agent,
            "liver_segmentation": self.segmentation_agent,
            "clinical_reasoning": self.clinical_reasoning_agent
        }

        print("=" * 80)
        print("LIVERAI MULTI-AGENT ORCHESTRATOR")
        print("=" * 80)

        for name, agent in self.agents.items():

            if agent is not None:
                print(
                    f"✓ {name:<25} "
                    f"{agent.__class__.__name__}"
                )
            else:
                print(
                    f"✗ {name:<25} NOT LOADED"
                )

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
    # GENERIC AGENT EXECUTION
    # ==========================================================

    def _run_agent(
        self,
        agent_name,
        agent,
        input_data
    ):

        self._log(
            f"[{agent_name.upper()}] Starting..."
        )

        # ------------------------------------------------------
        # Check agent
        # ------------------------------------------------------

        if agent is None:

            return {
                "agent": agent_name,
                "status": "not_loaded",
                "prediction": None,
                "error": "Agent is not loaded."
            }

        # ------------------------------------------------------
        # Check input
        # ------------------------------------------------------

        if input_data is None:

            self._log(
                f"[{agent_name.upper()}] "
                "No input → skipped."
            )

            return {
                "agent": agent_name,
                "status": "not_available",
                "prediction": None
            }

        # ------------------------------------------------------
        # Execute predict()
        # ------------------------------------------------------

        try:

            result = agent.predict(input_data)

            # --------------------------------------------------
            # Normalize result
            # --------------------------------------------------

            if result is None:

                result = {}

            if not isinstance(result, dict):

                result = {
                    "prediction": result
                }

            result = dict(result)

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
                f"[{agent_name.upper()}] ✗ ERROR: {e}"
            )

            return {
                "agent": agent_name,
                "status": "error",
                "prediction": None,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    # ==========================================================
    # AGENT 1 — FATTY LIVER
    # ==========================================================

    def run_fatty_liver(self, patient_data):

        return self._run_agent(
            "fatty_liver",
            self.fatty_agent,
            patient_data
        )

    # ==========================================================
    # AGENT 2 — FIBROSIS
    #
    # IMPORTANT:
    #
    # This receives patient_data.
    #
    # Your XGBoost model expects 9 features:
    #
    # age
    # male
    # weight
    # height
    # bmi
    # futime
    # days
    # test
    # value
    #
    # The FibrosisAgent itself is responsible for preparing
    # these features.
    # ==========================================================

    def run_fibrosis(self, patient_data):

        return self._run_agent(
            "fibrosis",
            self.fibrosis_agent,
            patient_data
        )

    # ==========================================================
    # AGENT 3 — CIRRHOSIS
    # ==========================================================

    def run_cirrhosis(self, patient_data):

        return self._run_agent(
            "cirrhosis",
            self.cirrhosis_agent,
            patient_data
        )

    # ==========================================================
    # AGENT 4 — TUMOR CLASSIFICATION
    #
    # Receives MRI/image data.
    # ==========================================================

    def run_tumor_classification(self, mri_image):

        return self._run_agent(
            "tumor_classification",
            self.tumor_agent,
            mri_image
        )

    # ==========================================================
    # AGENT 5 — LIVER SEGMENTATION
    #
    # Receives 3D liver volume.
    # ==========================================================

    def run_liver_segmentation(self, liver_volume):

        return self._run_agent(
            "liver_segmentation",
            self.segmentation_agent,
            liver_volume
        )

    # ==========================================================
    # RUN FIVE SPECIALIZED AGENTS
    # ==========================================================

    def run_specialized_agents(
        self,
        patient_data=None,
        mri_image=None,
        liver_volume=None
    ):

        self._log("")
        self._log("=" * 80)
        self._log("SPECIALIZED AGENTS")
        self._log("=" * 80)

        results = {}

        # ------------------------------------------------------
        # 1. FATTY LIVER
        # ------------------------------------------------------

        self._log("\n[1/5] Fatty Liver Agent")

        results["fatty_liver"] = (
            self.run_fatty_liver(
                patient_data
            )
        )

        # ------------------------------------------------------
        # 2. FIBROSIS
        # ------------------------------------------------------

        self._log("\n[2/5] Fibrosis Agent")

        results["fibrosis"] = (
            self.run_fibrosis(
                patient_data
            )
        )

        # ------------------------------------------------------
        # 3. CIRRHOSIS
        # ------------------------------------------------------

        self._log("\n[3/5] Cirrhosis Agent")

        results["cirrhosis"] = (
            self.run_cirrhosis(
                patient_data
            )
        )

        # ------------------------------------------------------
        # 4. TUMOR CLASSIFICATION
        # ------------------------------------------------------

        self._log("\n[4/5] Tumor Classification Agent")

        results["tumor_classification"] = (
            self.run_tumor_classification(
                mri_image
            )
        )

        # ------------------------------------------------------
        # 5. LIVER SEGMENTATION
        # ------------------------------------------------------

        self._log("\n[5/5] Liver Segmentation Agent")

        results["liver_segmentation"] = (
            self.run_liver_segmentation(
                liver_volume
            )
        )

        # ------------------------------------------------------
        # Save results
        # ------------------------------------------------------

        self.last_results = results

        self._log("")
        self._log("-" * 80)
        self._log("SPECIALIZED RESULTS COLLECTED")
        self._log("-" * 80)

        for name, result in results.items():

            status = result.get(
                "status",
                "unknown"
            )

            self._log(
                f"{name:<25} → {status}"
            )

        return results

    # ==========================================================
    # AGENT 6 — CLINICAL REASONING
    #
    # Receives results from ALL FIVE specialized agents.
    # ==========================================================

    def run_clinical_reasoning(
        self,
        agent_results
    ):

        self._log("")
        self._log("=" * 80)
        self._log("CLINICAL REASONING AGENT")
        self._log("=" * 80)

        try:

            # --------------------------------------------------
            # Pass ALL five agent results directly
            # --------------------------------------------------

            reasoning = (
                self.clinical_reasoning_agent.predict(
                    agent_results
                )
            )

            if reasoning is None:

                reasoning = {}

            if not isinstance(reasoning, dict):

                reasoning = {
                    "assessment": reasoning
                }

            reasoning.setdefault(
                "status",
                "success"
            )

            self._log(
                "✓ Clinical Reasoning completed"
            )

            return reasoning

        except Exception as e:

            self._log(
                f"✗ Clinical Reasoning ERROR: {e}"
            )

            return {
                "status": "error",
                "assessment": None,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    # ==========================================================
    # BUILD FINAL ASSESSMENT
    # ==========================================================

    def build_unified_assessment(
        self,
        agent_results,
        clinical_reasoning
    ):

        specialized_agents = [
            "fatty_liver",
            "fibrosis",
            "cirrhosis",
            "tumor_classification",
            "liver_segmentation"
        ]

        successful = 0
        errors = 0
        unavailable = 0

        for name in specialized_agents:

            result = agent_results.get(
                name,
                {}
            )

            status = result.get(
                "status"
            )

            if status == "success":
                successful += 1

            elif status == "error":
                errors += 1

            else:
                unavailable += 1

        if errors > 0:

            system_status = "partial"

        elif successful == 5:

            system_status = "completed"

        else:

            system_status = "partial"

        assessment = {

            "system":
                "LiverAI-MultiAgent",

            "timestamp":
                datetime.now().isoformat(),

            "specialized_agents":
                agent_results,

            "clinical_reasoning":
                clinical_reasoning,

            "coordination": {

                "total_agents": 6,

                "specialized_agents": 5,

                "clinical_reasoning_agent": 1,

                "successful_specialized_agents":
                    successful,

                "failed_specialized_agents":
                    errors,

                "unavailable_specialized_agents":
                    unavailable
            },

            "system_status":
                system_status
        }

        self.last_assessment = assessment

        return assessment

    # ==========================================================
    # MAIN PIPELINE
    # ==========================================================

    def predict(
        self,
        patient_data=None,
        mri_image=None,
        liver_volume=None
    ):

        self.execution_log = []

        self._log("")
        self._log("=" * 80)
        self._log("LIVERAI MULTI-AGENT ANALYSIS")
        self._log("=" * 80)

        # ======================================================
        # STEP 1 — SPECIALIZED AGENTS
        # ======================================================

        self._log(
            "\nSTEP 1/3 → Running 5 Specialized Agents"
        )

        agent_results = (
            self.run_specialized_agents(

                patient_data=patient_data,

                mri_image=mri_image,

                liver_volume=liver_volume
            )
        )

        # ======================================================
        # STEP 2 — CLINICAL REASONING
        # ======================================================

        self._log(
            "\nSTEP 2/3 → Coordinating Agent Results"
        )

        clinical_reasoning = (
            self.run_clinical_reasoning(
                agent_results
            )
        )

        # ======================================================
        # STEP 3 — FINAL ASSESSMENT
        # ======================================================

        self._log(
            "\nSTEP 3/3 → Building Unified Assessment"
        )

        final_assessment = (
            self.build_unified_assessment(

                agent_results,

                clinical_reasoning
            )
        )

        self._log("")
        self._log("=" * 80)
        self._log("✓ LIVERAI MULTI-AGENT ANALYSIS COMPLETED")
        self._log("=" * 80)

        return final_assessment

    # ==========================================================
    # GETTERS
    # ==========================================================

    def get_last_results(self):

        return self.last_results

    def get_last_assessment(self):

        return self.last_assessment

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
                    (
                        agent.__class__.__name__
                        if agent is not None
                        else None
                    )
            }

        return status
```
